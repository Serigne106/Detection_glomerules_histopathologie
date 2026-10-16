"""Lecture des images / masques, appariement des fichiers, encodage RLE."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_rgb(path) -> np.ndarray:
    """Lecture robuste en RGB uint8 (niveaux de gris, alpha, 16 bits acceptés)."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return to_rgb_uint8(image, bgr=True)


def to_rgb_uint8(image: np.ndarray, bgr: bool = False) -> np.ndarray:
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if bgr else cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if bgr else np.ascontiguousarray(image)


def read_instances(path) -> np.ndarray:
    """Masque d'instances int32 (0 = fond, k = glomérule k).

    Si le fichier est un simple masque binaire (0/1 ou 0/255), les instances sont
    reconstruites par composantes connexes.
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    values = np.unique(m)
    if len(values) <= 2 and values.max() in (1, 255):
        _, labels = cv2.connectedComponents((m > 0).astype(np.uint8), connectivity=8)
        return labels.astype(np.int32)
    return m.astype(np.int32)


def read_mask(path) -> np.ndarray:
    return read_instances(path) > 0


def list_pairs(directory, image_suffix=".png", mask_suffix="_mask.tif"):
    """Paires (image, masque) d'un dossier + liste des masques sans image."""
    pairs, orphans = [], []
    for mask_path in sorted(Path(directory).glob(f"*{mask_suffix}")):
        img_path = mask_path.with_name(mask_path.name.replace(mask_suffix, image_suffix))
        if img_path.exists():
            pairs.append((img_path, mask_path))
        else:
            orphans.append(mask_path.name)
    return pairs, orphans


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS and "_mask" not in p.stem)


def group_key(name: str, regex: str | None) -> str:
    """Identifiant de groupe (zone de lame) extrait du nom de fichier."""
    if regex:
        m = re.search(regex, name)
        if m:
            return m.group(1)
    return name


def rle_encode(mask) -> str:
    """Masque binaire (H, W) -> "début longueur ..." (positions 1-indexées, ligne par ligne)."""
    pixels = np.asarray(mask, dtype=np.uint8).flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle: str, shape) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if rle:
        s = np.asarray(rle.split(), dtype=np.int64)
        for start, length in zip(s[0::2] - 1, s[1::2]):
            mask[start:start + length] = 1
    return mask.reshape(shape).astype(bool)
