"""Sauvegarde / rechargement : un seul fichier suffit pour prédire (poids + config + réglages)."""
from __future__ import annotations

from pathlib import Path

import torch

from .inference import Predictor
from .models import build_model


def save_checkpoint(path, model, cfg: dict, history: dict | None = None, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "history": history or {}, "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    """-> (modèle en mode eval, dict complet du checkpoint)."""
    # weights_only=True : on ne désérialise que des tenseurs et des types simples (pas de code arbitraire)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["config"]["model"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def load_predictor(path, device="cpu", **kwargs):
    """-> (Predictor, réglages d'inférence de déploiement ou None, checkpoint)."""
    model, ckpt = load_checkpoint(path, device)
    cfg, meta = ckpt["config"], ckpt.get("meta", {})
    predictor = Predictor(model, scale=cfg["input"]["scale"], normalization=cfg["input"]["normalization"],
                          device=device, target_diameter=meta.get("target_diameter"), **kwargs)
    return predictor, meta.get("inference"), ckpt
