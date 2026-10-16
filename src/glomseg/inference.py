"""Prédiction sur images de taille quelconque : TTA, multi-échelle, échelle automatique,
inférence par tuiles (grandes régions de lame) et carte d'incertitude."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import normalize_image, to_working_scale
from .postprocess import binarize, label_instances

# 8 symétries du carré : 4 rotations x miroir
D4 = [(k, flip) for flip in (False, True) for k in range(4)]

# Modes de prédiction comparés à l'évaluation
MODES: dict[str, dict] = {
    "simple": dict(tta=False, zooms=(1.0,)),
    "tta8": dict(tta=True, zooms=(1.0,)),
    "tta8_ms_centered": dict(tta=True, zooms=(0.8, 1.0, 1.25)),
    "tta8_ms_down": dict(tta=True, zooms=(0.6, 0.8, 1.0)),
    # échelle estimée image par image, puis TTA 8 symétries à cette échelle
    "auto": dict(tta=True, zooms=(1.0,), auto=True),
}


def _ramp_window(h, w, overlap):
    """Poids de fusion des tuiles : 1 au centre, décroissance linéaire sur les bords."""
    def ramp(n):
        r = np.ones(n, dtype=np.float32)
        if overlap > 0:
            edge = np.linspace(1.0 / overlap, 1.0, min(overlap, n), dtype=np.float32)
            r[:len(edge)] = np.minimum(r[:len(edge)], edge)
            r[n - len(edge):] = np.minimum(r[n - len(edge):], edge[::-1])
        return r
    return np.outer(ramp(h), ramp(w))


class Predictor:
    """Encapsule un réseau + ses réglages d'entrée. Toutes les sorties sont à la taille d'origine."""

    def __init__(self, model, scale=0.375, normalization="fixed", device="cpu",
                 tile_size=512, tile_overlap=64, target_diameter=None, pad_multiple=32):
        self.model = model.to(device).eval()
        self.scale = scale
        self.normalization = normalization
        self.device = torch.device(device)
        self.tile_size = tile_size              # en px de la résolution de travail
        self.tile_overlap = tile_overlap
        self.target_diameter = target_diameter  # diamètre médian des glomérules d'entraînement (px de travail)
        self.pad_multiple = pad_multiple

    # ------------------------------------------------------------------ réseau
    @torch.no_grad()
    def _net(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilités pour un tenseur (1, 3, H, W) ; pad à un multiple de 32 (encodeurs smp)."""
        h, w = x.shape[2:]
        ph, pw = (-h) % self.pad_multiple, (-w) % self.pad_multiple
        if ph or pw:
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return torch.sigmoid(self.model(x))[..., :h, :w]

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> np.ndarray:
        """Inférence directe ou par tuiles chevauchantes pour les grandes images."""
        h, w = x.shape[2:]
        t = self.tile_size
        if h <= t and w <= t:
            return self._net(x)[0, 0].cpu().numpy()
        stride = t - self.tile_overlap
        ys = list(range(0, max(h - t, 0) + 1, stride))
        xs = list(range(0, max(w - t, 0) + 1, stride))
        if ys[-1] + t < h:
            ys.append(h - t)
        if xs[-1] + t < w:
            xs.append(w - t)
        acc = np.zeros((h, w), dtype=np.float32)
        norm = np.zeros((h, w), dtype=np.float32)
        for y0 in ys:
            for x0 in xs:
                tile = x[..., y0:y0 + t, x0:x0 + t]
                p = self._net(tile)[0, 0].cpu().numpy()
                win = _ramp_window(*p.shape, self.tile_overlap)
                acc[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] += p * win
                norm[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] += win
        return acc / np.maximum(norm, 1e-6)

    # --------------------------------------------------------------- prédiction
    @torch.no_grad()
    def predict(self, image_rgb, tta=False, zooms=(1.0,), return_std=False):
        """Image RGB uint8 (H, W, 3) -> probabilités (H, W) [et écart-type entre passes].

        L'écart-type entre les passes (symétries x échelles) sert de carte d'incertitude :
        là où les versions de l'image ne sont pas d'accord, le modèle n'est pas sûr de lui.
        """
        h, w = image_rgb.shape[:2]
        s1 = np.zeros((h, w), dtype=np.float32)
        s2 = np.zeros((h, w), dtype=np.float32)
        n = 0
        for z in zooms:
            work = to_working_scale(image_rgb, scale=self.scale * z)
            x = torch.from_numpy(normalize_image(work, self.normalization)).unsqueeze(0).to(self.device)
            for k, flip in (D4 if tta else D4[:1]):
                xi = torch.rot90(x, k, dims=(2, 3))
                xi = torch.flip(xi, dims=(3,)) if flip else xi
                p = torch.from_numpy(self._forward(xi))[None, None]
                p = torch.flip(p, dims=(3,)) if flip else p
                p = torch.rot90(p, -k, dims=(2, 3))[0, 0].numpy()
                p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
                s1 += p
                s2 += p * p
                n += 1
        mean = s1 / n
        if not return_std:
            return mean
        std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
        return mean, std

    def estimate_zoom(self, image_rgb, threshold=0.5, min_area=1000, n_iter=2, bounds=(0.4, 2.5)) -> float:
        """Échelle automatique : on ajuste le zoom pour que les glomérules détectés aient la taille
        de ceux vus à l'entraînement (généralise la correction manuelle 0.6-1.0 du notebook).

        Itératif : prédiction -> diamètre médian des objets -> correction du zoom.
        Renvoie 1.0 si aucun objet n'est détecté ou si le diamètre cible est inconnu.
        """
        if not self.target_diameter or not np.isfinite(self.target_diameter):
            return 1.0
        zoom = 1.0
        for _ in range(n_iter):
            prob = self.predict(image_rgb, tta=False, zooms=(zoom,))
            labels = label_instances(binarize(prob, threshold, True, min_area))
            areas = np.bincount(labels.ravel())[1:]
            # les fragments (< 1/4 de l'aire attendue d'un glomérule) ne disent rien sur l'échelle
            expected = np.pi / 4 * (self.target_diameter / (self.scale * zoom)) ** 2
            areas = areas[areas >= 0.25 * expected]
            if len(areas) == 0:
                break
            # médiane pondérée par l'aire : les gros objets bien segmentés comptent davantage
            diams = np.sqrt(4 * np.sort(areas) / np.pi)
            cum = np.cumsum(np.sort(areas)) / areas.sum()
            d_work = float(diams[np.searchsorted(cum, 0.5)]) * self.scale * zoom
            new_zoom = float(np.clip(zoom * self.target_diameter / d_work, *bounds))
            if abs(np.log(new_zoom / zoom)) < 0.05:
                zoom = new_zoom
                break
            zoom = new_zoom
        return zoom

    def predict_mode(self, image_rgb, mode: str | dict, return_std=False, return_info=False):
        """Prédiction selon un mode nommé (voir ``MODES``) ou un dict de réglages."""
        cfg = dict(MODES[mode]) if isinstance(mode, str) else dict(mode)
        zooms = tuple(cfg.get("zooms", (1.0,)))
        base_zoom = 1.0
        if cfg.get("auto"):
            base_zoom = self.estimate_zoom(image_rgb, min_area=cfg.get("min_area", 1000))
            zooms = tuple(base_zoom * z for z in zooms)
        t0 = time.perf_counter()
        out = self.predict(image_rgb, tta=cfg.get("tta", False), zooms=zooms, return_std=return_std)
        if return_info:
            info = {"zoom": base_zoom, "zooms": zooms, "seconds": time.perf_counter() - t0}
            return out, info
        return out


def full_pipeline(predictor: Predictor, image_rgb, settings: dict, split_touching=False):
    """Image -> (probas, incertitude, masque final, étiquettes d'instances, infos).

    ``settings`` = {"mode": nom dans MODES, "threshold", "postprocess", "min_area"}.
    """
    mode = dict(MODES[settings["mode"]], min_area=settings["min_area"])
    (prob, std), info = predictor.predict_mode(image_rgb, mode, return_std=True, return_info=True)
    mask = binarize(prob, settings["threshold"], settings["postprocess"], settings["min_area"])
    labels = label_instances(mask, split_touching=split_touching)
    return prob, std, mask, labels, info
