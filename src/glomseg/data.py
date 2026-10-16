"""Datasets, normalisation, découpage réglage / test."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import Augmenter
from .io import group_key, list_pairs, read_instances, read_rgb

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_working_scale(image, mask=None, scale=0.375):
    h, w = image.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if mask is None:
        return image
    mask = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5
    return image, mask


def normalize_image(image, normalization="fixed"):
    """uint8 (H, W, 3) -> float32 (3, H, W)."""
    x = image.astype(np.float32) / 255.0
    if normalization == "imagenet":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    else:
        x = (x - 0.5) / 0.25
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def collect_pairs(root, dirs, image_suffix=".png", mask_suffix="_mask.tif"):
    pairs = []
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        p, _ = list_pairs(Path(root) / d, image_suffix, mask_suffix)
        pairs += p
    return pairs


class AugmentedDataset(Dataset):
    """Chaque accès fabrique une nouvelle image augmentée (jeu virtuellement infini)."""

    def __init__(self, pairs, cfg: dict, seed: int = 0):
        self.pairs = pairs
        self.scale = cfg["input"]["scale"]
        self.normalization = cfg["input"]["normalization"]
        self.samples_per_image = cfg["augment"]["samples_per_image"]
        self.augmenter = Augmenter(cfg["augment"], cfg["input"]["crop_size"])
        self.rng = np.random.default_rng(seed)
        self.images, self.masks, self.fg_coords = [], [], []
        for img_path, mask_path in pairs:
            image, mask = to_working_scale(read_rgb(img_path), read_instances(mask_path) > 0, self.scale)
            self.images.append(image)
            self.masks.append(mask)
            self.fg_coords.append(np.argwhere(mask))

    def __len__(self):
        return len(self.pairs) * self.samples_per_image

    def __getitem__(self, idx):
        i = idx % len(self.pairs)
        image, mask = self.augmenter(self.images[i], self.masks[i], self.fg_coords[i], self.rng)
        x = torch.from_numpy(normalize_image(image, self.normalization))
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y


def worker_init_fn(worker_id: int) -> None:
    """Chaque worker reçoit son propre générateur.

    Sans cela, avec ``num_workers > 0``, tous les workers héritent du même état numpy
    et produisent les MÊMES augmentations (bug classique de PyTorch + numpy).
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


class EvalSet:
    """Images entières à leur taille d'origine, avec leurs masques d'instances."""

    def __init__(self, pairs, group_regex=None):
        self.pairs = pairs
        self.names = [p.stem for p, _ in pairs]
        self.images = [read_rgb(p) for p, _ in pairs]
        self.instances = [read_instances(m) for _, m in pairs]
        self.masks = [m > 0 for m in self.instances]
        self.groups = [group_key(p.name, group_regex) for p, _ in pairs]

    def __len__(self):
        return len(self.pairs)


def make_folds(groups: list[str], max_folds: int = 3, seed: int = 0):
    """Plis de cross-fitting : chaque image est testée exactement une fois, avec des réglages
    choisis SANS elle.

    - au moins 2 groupes (zones de lame) : leave-one-group-out ;
    - un seul groupe : K plis aléatoires par image.
    Renvoie une liste de (indices_réglage, indices_test).
    """
    n = len(groups)
    uniq = sorted(set(groups))
    if len(uniq) >= 2:
        return [([i for i in range(n) if groups[i] != g], [i for i in range(n) if groups[i] == g]) for g in uniq]
    k = max(2, min(max_folds, n))
    order = np.random.default_rng(seed).permutation(n)
    chunks = np.array_split(order, k)
    return [(sorted(int(i) for i in np.concatenate([c for j, c in enumerate(chunks) if j != f])),
             sorted(int(i) for i in chunks[f])) for f in range(k)]


def median_diameter(instance_masks, scale=1.0) -> float:
    """Diamètre équivalent médian (sqrt(4A/pi)) des glomérules, à l'échelle donnée."""
    diams = []
    for m in instance_masks:
        ids, counts = np.unique(m[m > 0], return_counts=True)
        diams += list(np.sqrt(4 * counts / np.pi))
    return float(np.median(diams) * scale) if diams else float("nan")
