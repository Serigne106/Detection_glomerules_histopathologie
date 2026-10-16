"""Métriques : Dice (pixel), détection par instance (appariement IoU), intervalles bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def dice(pred, true) -> float:
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    if pred.shape != true.shape:
        raise ValueError(f"Formes différentes : {pred.shape} vs {true.shape}")
    denom = pred.sum() + true.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, true).sum() / denom)


def _relabel(labels):
    ids = np.unique(labels[labels > 0])
    remap = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int64)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[labels], len(ids)


def overlap_matrix(gt_labels, pred_labels):
    """Intersections (n_gt x n_pred), aires GT et prédites, en un seul passage (bincount)."""
    g, ng = _relabel(np.asarray(gt_labels))
    p, npred = _relabel(np.asarray(pred_labels))
    inter = np.bincount((g * (npred + 1) + p).ravel(), minlength=(ng + 1) * (npred + 1))
    inter = inter.reshape(ng + 1, npred + 1)
    gt_area = inter.sum(axis=1)[1:]
    pred_area = inter.sum(axis=0)[1:]
    return inter[1:, 1:], gt_area, pred_area


def instance_metrics(gt_labels, pred_labels, iou_threshold=0.5) -> dict:
    """Appariement 1-1 (hongrois) des glomérules prédits et réels, IoU >= seuil.

    Catégories d'erreurs (pour l'analyse) :
    - FN ``missed`` : glomérule réel quasiment pas touché (IoU max < 0.1) ;
    - FN ``poorly_segmented`` : touché mais IoU < seuil ;
    - ``merged`` : glomérules réels fusionnés en un seul objet prédit ;
    - FP ``spurious`` : objet prédit sans glomérule dessous (IoU max < 0.1).
    """
    inter, gt_area, pred_area = overlap_matrix(gt_labels, pred_labels)
    ng, npred = len(gt_area), len(pred_area)
    out = {"n_gt": ng, "n_pred": npred, "tp": 0, "fp": npred, "fn": ng, "matched_iou": [],
           "fn_missed": ng, "fn_poorly_segmented": 0, "fp_spurious": npred, "merged": 0}
    if ng == 0 or npred == 0:
        return _finish(out)
    union = gt_area[:, None] + pred_area[None, :] - inter
    iou = inter / np.maximum(union, 1)
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] >= iou_threshold
    rows, cols = rows[keep], cols[keep]
    tp = len(rows)
    out.update(tp=tp, fp=npred - tp, fn=ng - tp, matched_iou=iou[rows, cols].tolist())
    gt_unmatched = np.setdiff1d(np.arange(ng), rows)
    pred_unmatched = np.setdiff1d(np.arange(npred), cols)
    max_iou_gt = iou.max(axis=1)
    max_iou_pred = iou.max(axis=0)
    out["fn_missed"] = int((max_iou_gt[gt_unmatched] < 0.1).sum())
    out["fn_poorly_segmented"] = int(len(gt_unmatched) - out["fn_missed"])
    out["fp_spurious"] = int((max_iou_pred[pred_unmatched] < 0.1).sum())
    covered = inter / np.maximum(gt_area[:, None], 1) >= 0.5      # GT couvert à 50 % par l'objet prédit
    out["merged"] = int(covered.sum(axis=0)[covered.sum(axis=0) >= 2].sum())
    return _finish(out)


def _finish(out):
    tp, fp, fn = out["tp"], out["fp"], out["fn"]
    out["precision"] = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    out["recall"] = tp / (tp + fn) if tp + fn else 1.0
    pr = out["precision"] + out["recall"]
    out["f1"] = 2 * out["precision"] * out["recall"] / pr if pr else 0.0
    out["mean_matched_iou"] = float(np.mean(out["matched_iou"])) if out["matched_iou"] else float("nan")
    out["count_error"] = out["n_pred"] - out["n_gt"]
    return out


def micro_f1(rows) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def bootstrap_ci(rows, statistic, n_boot=2000, alpha=0.05, seed=0):
    """IC percentile par rééchantillonnage des IMAGES (l'unité statistique indépendante)."""
    rows = list(rows)
    if len(rows) < 2:
        v = statistic(rows)
        return v, v
    rng = np.random.default_rng(seed)
    stats = [statistic([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def summarize(rows, n_boot=2000) -> dict:
    """Résumé d'un ensemble d'images : Dice moyen ± écart-type, IC 95 %, F1 instance (micro), comptage."""
    dices = [r["dice"] for r in rows]
    mean_dice = lambda rs: float(np.mean([r["dice"] for r in rs]))
    f1 = lambda rs: micro_f1(rs)["f1"]
    micro = micro_f1(rows)
    return {
        "n_images": len(rows),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
        "dice_median": float(np.median(dices)),
        "dice_min": float(np.min(dices)),
        "dice_ci95": bootstrap_ci(rows, mean_dice, n_boot),
        "instance_precision": micro["precision"],
        "instance_recall": micro["recall"],
        "instance_f1": micro["f1"],
        "instance_f1_ci95": bootstrap_ci(rows, f1, n_boot),
        "count_mae": float(np.mean([abs(r["count_error"]) for r in rows])),
        "n_gt": int(sum(r["n_gt"] for r in rows)),
        "fn_missed": int(sum(r["fn_missed"] for r in rows)),
        "fn_poorly_segmented": int(sum(r["fn_poorly_segmented"] for r in rows)),
        "fp_spurious": int(sum(r["fp_spurious"] for r in rows)),
        "merged": int(sum(r["merged"] for r in rows)),
    }
