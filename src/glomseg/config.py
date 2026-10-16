"""Chargement des configurations YAML.

Une config peut hériter d'une autre avec la clé ``_base_`` (chemin relatif au fichier).
Tout ce qui n'est pas précisé prend la valeur de ``DEFAULTS``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "unet_scratch",
    "data": {
        "root": "./Set02_Glom_train_clean",
        "train_dirs": ["train03"],          # dossiers d'entraînement
        "eval_dir": "train_AKI",            # domaine jamais vu : réglage + test (cross-fitting)
        "group_regex": r"svs_(\d+)_",       # regroupe les imagettes d'une même zone de lame
        "image_suffix": ".png",
        "mask_suffix": "_mask.tif",
    },
    "input": {
        "scale": 0.375,                     # 1024 px -> 384 px
        "crop_size": 192,
        "normalization": "fixed",           # "fixed" (de zéro) ou "imagenet" (encodeur pré-entraîné)
    },
    "augment": {
        "samples_per_image": 8,
        "zoom_range": [0.55, 1.8],
        "pos_crop_prob": 0.6,
        "elastic_prob": 0.3,
        "color": True,
        "quality": True,
    },
    "model": {
        "arch": "unet_scratch",             # "unet_scratch" ou "smp_unet"
        "base_channels": 16,
        "depth": 5,
        "dropout": 0.2,
        "encoder": "resnet34",              # smp_unet uniquement
        "encoder_weights": "imagenet",      # smp_unet uniquement
    },
    "train": {
        "batch_size": 8,
        "epochs": 60,
        "lr": 2.0e-3,
        "weight_decay": 1.0e-4,
        "ema_decay": 0.99,
        "seed": 42,
        "num_workers": 0,
        # "last" : on garde le modèle de la dernière epoch -> aucune sélection sur le domaine de test.
        # "best" : meilleur Dice de validation (fuite d'information, le score devient optimiste).
        "checkpoint": "last",
        "val_every": 5,                     # 0 = pas de suivi de validation pendant l'entraînement
        "mlflow": False,
    },
    "eval": {
        "thresholds": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "modes": ["simple", "tta8", "tta8_ms_centered", "tta8_ms_down", "auto"],
        "min_area": 1000,                   # px de l'image d'origine
        "iou_threshold": 0.5,
        "split_touching": False,
        "n_bootstrap": 2000,
        "n_gallery": 4,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = deep_merge(cfg, _load_with_base(Path(path)))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def _load_with_base(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("_base_", None)
    if base is None:
        return raw
    return deep_merge(_load_with_base((path.parent / base).resolve()), raw)


def parse_overrides(items: list[str] | None) -> dict:
    """``["train.epochs=5", "model.arch=smp_unet"]`` -> dict imbriqué (valeurs parsées en YAML)."""
    out: dict = {}
    for item in items or []:
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
