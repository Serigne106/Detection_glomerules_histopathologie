"""Données synthétiques : tissu rose texturé avec des "glomérules" ronds plus sombres."""
import cv2
import numpy as np
import pytest


def synthetic_sample(rng, size=256, n_objects=3, radius=(18, 28)):
    image = np.full((size, size, 3), (225, 170, 200), np.uint8)
    noise = rng.normal(0, 12, (size, size, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    inst = np.zeros((size, size), np.uint8)
    k = 0
    for _ in range(n_objects * 5):
        if k == n_objects:
            break
        r = int(rng.integers(*radius))
        cx, cy = (int(v) for v in rng.integers(r + 2, size - r - 2, 2))
        if inst[cy - r:cy + r, cx - r:cx + r].any():
            continue
        k += 1
        cv2.circle(inst, (cx, cy), r, k, -1)
        cv2.circle(image, (cx, cy), r, (150, 60, 140), -1)
        cv2.circle(image, (cx, cy), r, (250, 240, 245), 2)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image, inst


@pytest.fixture
def synthetic_dir(tmp_path):
    """Arborescence data/{train,eval} au format du challenge (png + _mask.tif)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    for split, n in (("train", 6), ("eval", 4)):
        d = root / split
        d.mkdir(parents=True)
        for i in range(n):
            image, inst = synthetic_sample(rng)
            name = f"slide.svs_{1000 * (i % 2)}_{i}"
            cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / f"{name}_mask.tif"), inst)
    return root


@pytest.fixture
def tiny_cfg(synthetic_dir):
    from glomseg.config import load_config
    return load_config(overrides={
        "name": "test",
        "data": {"root": str(synthetic_dir), "train_dirs": ["train"], "eval_dir": "eval"},
        "input": {"scale": 0.5, "crop_size": 64},
        "augment": {"samples_per_image": 2},
        "model": {"base_channels": 4, "depth": 3},
        "train": {"epochs": 2, "batch_size": 4, "val_every": 1},
        "eval": {"thresholds": [0.4, 0.5], "modes": ["simple", "tta8", "auto"], "min_area": 50,
                 "n_bootstrap": 50, "n_gallery": 1},
    })
