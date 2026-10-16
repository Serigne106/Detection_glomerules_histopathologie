"""Augmentations de données (reprises du notebook, paramétrées par la config).

Géométrie : appliquée à l'image ET au masque. Couleur / qualité : image seule.
Chaque bloc peut être désactivé dans la config pour les ablations.
"""
from __future__ import annotations

import cv2
import numpy as np

# Déconvolution de couleur de Ruifrok & Johnston (2001) ; perturbation par colorant (Tellez et al., 2018)
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]], dtype=np.float32)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)


def random_affine_crop(image, mask, fg_coords, crop, zoom_range, pos_crop_prob, rng):
    """Centre (sur un glomérule ou non), zoom log-uniforme, rotation quelconque, miroir : un seul warpAffine."""
    h, w = mask.shape
    if len(fg_coords) and rng.random() < pos_crop_prob:
        cy, cx = fg_coords[rng.integers(len(fg_coords))]
        cy, cx = cy + rng.uniform(-crop / 3, crop / 3), cx + rng.uniform(-crop / 3, crop / 3)
    else:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    lo, hi = zoom_range
    zoom = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if hi > lo else float(lo)
    angle = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, zoom)
    M[:, 2] += (crop / 2 - cx, crop / 2 - cy)
    image = cv2.warpAffine(image, M, (crop, crop), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask.astype(np.uint8), M, (crop, crop), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def elastic_deform(image, mask, rng, alpha=120.0, sigma=8.0):
    h, w = mask.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = gx + dx, gy + dy
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def hed_jitter(image, rng, sigma=0.35, bias=0.03):
    od = -np.log((image.astype(np.float32) + 1.0) / 256.0)
    hed = od.reshape(-1, 3) @ HED_FROM_RGB
    hed = hed * rng.uniform(1 - sigma, 1 + sigma, 3) + rng.uniform(-bias, bias, 3)
    od = (hed @ RGB_FROM_HED).reshape(image.shape)
    return np.clip(np.exp(-od) * 256.0 - 1.0, 0, 255).astype(np.uint8)


def hsv_jitter(image, rng):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-15, 15)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.8, 1.2)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def contrast_gamma(image, rng):
    x = image.astype(np.float32) / 255.0
    mean = x.mean()
    x = (x - mean) * rng.uniform(0.7, 1.3) + mean + rng.uniform(-0.08, 0.08)
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.5)
    return (x * 255).astype(np.uint8)


def color_augment(image, rng):
    if rng.random() < 0.8:
        image = hed_jitter(image, rng)
    if rng.random() < 0.7:
        image = hsv_jitter(image, rng)
    if rng.random() < 0.7:
        image = contrast_gamma(image, rng)
    if rng.random() < 0.1:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if rng.random() < 0.05:
        image = np.ascontiguousarray(image[..., rng.permutation(3)])
    return image


def quality_augment(image, rng):
    if rng.random() < 0.25:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.25:
        noise = rng.normal(0, rng.uniform(2, 10), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        if ok:
            image = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


class Augmenter:
    """Chaîne complète, pilotée par la section ``augment`` de la config."""

    def __init__(self, aug_cfg: dict, crop_size: int):
        self.cfg = aug_cfg
        self.crop = crop_size

    def __call__(self, image, mask, fg_coords, rng: np.random.Generator):
        c = self.cfg
        image, mask = random_affine_crop(image, mask, fg_coords, self.crop, tuple(c["zoom_range"]),
                                         c["pos_crop_prob"], rng)
        if rng.random() < c["elastic_prob"]:
            image, mask = elastic_deform(image, mask, rng)
        if c["color"]:
            image = color_augment(image, rng)
        if c["quality"]:
            image = quality_augment(image, rng)
        return image, mask
