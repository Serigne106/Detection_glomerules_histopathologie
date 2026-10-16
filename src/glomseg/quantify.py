"""Quantification : ce qu'un pathologiste veut lire (nombre, tailles, confiance par glomérule)."""
from __future__ import annotations

import numpy as np


def instances_table(labels, prob=None, std=None, mpp: float | None = None, uncertain_std=0.15) -> list[dict]:
    """Une ligne par glomérule détecté.

    ``mpp`` (microns par pixel), si connu, donne les surfaces en µm².
    ``uncertain`` signale les objets où les passes de TTA divergent (à vérifier par un humain).
    """
    rows = []
    for k in range(1, int(labels.max()) + 1):
        obj = labels == k
        area = int(obj.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(obj)
        row = {
            "id": k,
            "x": int(xs.min()), "y": int(ys.min()),
            "width": int(xs.max() - xs.min() + 1), "height": int(ys.max() - ys.min() + 1),
            "centroid_x": round(float(xs.mean()), 1), "centroid_y": round(float(ys.mean()), 1),
            "area_px": area,
            "eq_diameter_px": round(float(np.sqrt(4 * area / np.pi)), 1),
        }
        if mpp:
            row["area_um2"] = round(area * mpp ** 2, 1)
            row["eq_diameter_um"] = round(row["eq_diameter_px"] * mpp, 1)
        if prob is not None:
            row["mean_prob"] = round(float(prob[obj].mean()), 3)
        if std is not None:
            row["mean_uncertainty"] = round(float(std[obj].mean()), 3)
            row["uncertain"] = bool(row["mean_uncertainty"] > uncertain_std)
        rows.append(row)
    return rows


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n_glomeruli": 0}
    areas = np.array([r["area_px"] for r in rows])
    out = {
        "n_glomeruli": len(rows),
        "median_area_px": float(np.median(areas)),
        "median_eq_diameter_px": float(np.median([r["eq_diameter_px"] for r in rows])),
    }
    if "uncertain" in rows[0]:
        out["n_uncertain"] = int(sum(r["uncertain"] for r in rows))
    if "area_um2" in rows[0]:
        out["median_area_um2"] = float(np.median([r["area_um2"] for r in rows]))
    return out
