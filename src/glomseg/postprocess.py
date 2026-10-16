"""Post-traitement : nettoyage du masque et passage pixels -> instances."""
from __future__ import annotations

import cv2
import numpy as np


def clean_mask(mask, min_area: int = 1000) -> np.ndarray:
    """Supprime les taches < ``min_area`` px et bouche les trous de chaque objet."""
    mask = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def binarize(prob, threshold: float, postprocess: bool, min_area: int) -> np.ndarray:
    pred = prob > threshold
    return clean_mask(pred, min_area) if postprocess else pred


def label_instances(mask, split_touching: bool = False, peak_ratio: float = 0.6) -> np.ndarray:
    """Masque binaire -> étiquettes int32 (0 = fond).

    ``split_touching`` : sépare les glomérules collés par ligne de partage des eaux sur la
    carte de distance (un marqueur par "cœur" d'objet). Sans effet sur le Dice, mais
    améliore le comptage et les métriques par instance.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if not split_touching or n <= 1:
        return labels.astype(np.int32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cores = np.zeros_like(mask)
    for k in range(1, n):
        comp = labels == k
        cores[comp & (dist >= peak_ratio * dist[comp].max())] = 1
    n_cores, markers = cv2.connectedComponents(cores, connectivity=8)
    if n_cores - 1 <= n - 1:          # aucun objet à séparer
        return labels.astype(np.int32)
    markers = markers.astype(np.int32) + 1
    markers[mask == 0] = 1            # fond = marqueur 1
    markers[(mask == 1) & (cores == 0)] = 0   # zone à attribuer
    rgb = cv2.cvtColor((dist / max(dist.max(), 1e-6) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(255 - rgb, markers)
    out = markers - 1
    out[(out < 1) | (mask == 0)] = 0
    # les lignes de partage (-1) sont rendues à l'objet voisin le plus proche
    boundary = (markers == -1) & (mask == 1)
    if boundary.any():
        dilated = cv2.dilate(out.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        out[boundary] = dilated[boundary]
    # renumérotation compacte 1..K
    ids = np.unique(out[out > 0])
    remap = np.zeros(out.max() + 1, dtype=np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[out]
