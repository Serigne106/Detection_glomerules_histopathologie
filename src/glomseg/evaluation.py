"""Protocole d'évaluation honnête.

Problème du notebook d'origine : le même jeu (train_AKI) servait à choisir le mode de prédiction,
le seuil, le post-traitement ET à annoncer le score -> score optimiste.

Ici, CROSS-FITTING : les images d'évaluation sont découpées par zone de lame (groupes). Pour
chaque pli, les réglages sont choisis sur les autres zones puis appliqués à la zone mise de côté.
Chaque image est donc prédite avec des réglages choisis SANS elle. On rapporte aussi le score
"optimiste" (réglé et mesuré sur les mêmes images) pour montrer l'écart.
"""
from __future__ import annotations

import csv
import json
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from .data import EvalSet, make_folds
from .inference import MODES, Predictor
from .metrics import dice, instance_metrics, summarize
from .postprocess import binarize, label_instances
from .viz import gallery_row


def predict_all(predictor: Predictor, eval_set: EvalSet, modes, min_area=1000, log=print):
    """Prédit chaque image dans chaque mode. Mesure le temps par image."""
    probs, stds, seconds, zooms = {}, {}, {}, {}
    for mode in modes:
        cfg = dict(MODES[mode], min_area=min_area)
        probs[mode], stds[mode], seconds[mode], zooms[mode] = [], [], [], []
        for image in eval_set.images:
            t0 = time.perf_counter()
            (p, s), info = predictor.predict_mode(image, cfg, return_std=True, return_info=True)
            seconds[mode].append(time.perf_counter() - t0)          # inclut l'estimation d'échelle
            probs[mode].append(p.astype(np.float16))
            stds[mode].append(s.astype(np.float16))
            zooms[mode].append(info["zoom"])
        log(f"  {mode:<18} {np.mean(seconds[mode]):6.2f} s/image")
    return probs, stds, seconds, zooms


def dice_grid(probs, masks, thresholds, min_area):
    """Dice de chaque image pour chaque réglage (mode, seuil, post-traitement)."""
    grid = {}
    for mode, t, post in product(probs, thresholds, (False, True)):
        grid[(mode, float(t), post)] = np.array(
            [dice(binarize(p.astype(np.float32), t, post, min_area), m) for p, m in zip(probs[mode], masks)])
    return grid


def best_setting(grid, idx, modes=None):
    keys = [k for k in grid if modes is None or k[0] in modes]
    # à égalité, on préfère le mode listé en premier (le plus simple) puis le seuil le plus proche de 0.5
    return max(keys, key=lambda k: (round(float(grid[k][idx].mean()), 6), -abs(k[1] - 0.5)))


def cross_fit(grid, folds, modes=None):
    """-> réglage choisi pour chaque image (choisi sans elle)."""
    n = len(next(iter(grid.values())))
    chosen = [None] * n
    for tune_idx, test_idx in folds:
        k = best_setting(grid, tune_idx, modes)
        for i in test_idx:
            chosen[i] = k
    return chosen


def per_image_rows(eval_set, probs, chosen, min_area, iou_threshold, split_touching):
    rows = []
    for i, (mode, t, post) in enumerate(chosen):
        pred = binarize(probs[mode][i].astype(np.float32), t, post, min_area)
        inst = instance_metrics(eval_set.instances[i], label_instances(pred, split_touching), iou_threshold)
        inst.pop("matched_iou")
        rows.append({"image": eval_set.names[i], "group": eval_set.groups[i], "mode": mode, "threshold": t,
                     "postprocess": post, "dice": dice(pred, eval_set.masks[i]), **inst})
    return rows


def evaluate(predictor: Predictor, eval_set: EvalSet, cfg: dict, out_dir, log=print) -> dict:
    ec = cfg["eval"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = list(ec["modes"])
    min_area = ec["min_area"]
    log(f"Prédiction de {len(eval_set)} images dans {len(modes)} modes")
    probs, stds, seconds, zooms = predict_all(predictor, eval_set, modes, min_area, log)
    grid = dice_grid(probs, eval_set.masks, ec["thresholds"], min_area)
    folds = make_folds(eval_set.groups)
    log(f"Cross-fitting : {len(folds)} plis ({'par zone de lame' if len(set(eval_set.groups)) > 1 else 'par image'})")
    all_idx = list(range(len(eval_set)))
    args = (min_area, ec["iou_threshold"], ec["split_touching"])

    # 1) protocole honnête : réglages choisis sans l'image testée
    honest_rows = per_image_rows(eval_set, probs, cross_fit(grid, folds), *args)
    # 2) protocole du notebook (optimiste) : réglé et mesuré sur les mêmes images
    deploy = best_setting(grid, all_idx)
    optimistic_rows = per_image_rows(eval_set, probs, [deploy] * len(eval_set), *args)
    # 3) chaque mode de prédiction, réglé honnêtement (seuil / post-traitement), + latence
    by_mode = {}
    for mode in modes:
        rows = per_image_rows(eval_set, probs, cross_fit(grid, folds, [mode]), *args)
        by_mode[mode] = {**summarize(rows, ec["n_bootstrap"]),
                         "seconds_per_image": float(np.mean(seconds[mode])),
                         "mean_auto_zoom": float(np.mean(zooms[mode]))}

    results = {
        "name": cfg["name"],
        "honest": summarize(honest_rows, ec["n_bootstrap"]),
        "optimistic": summarize(optimistic_rows, ec["n_bootstrap"]),
        "by_mode": by_mode,
        "deploy_settings": {"mode": deploy[0], "threshold": deploy[1], "postprocess": deploy[2],
                            "min_area": min_area},
        "folds": [{"tune": [eval_set.names[i] for i in a], "test": [eval_set.names[i] for i in b]}
                  for a, b in folds],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    _write_csv(out_dir / "per_image.csv", honest_rows)
    _write_threshold_curves(out_dir / "threshold_curves.csv", grid)
    _write_gallery(out_dir, eval_set, probs, stds, honest_rows, min_area, ec["n_gallery"])
    (out_dir / "report.md").write_text(render_report(results, honest_rows), encoding="utf-8")
    log(f"Dice honnête {results['honest']['dice_mean']:.4f} "
        f"[IC95 {results['honest']['dice_ci95'][0]:.3f}-{results['honest']['dice_ci95'][1]:.3f}] | "
        f"optimiste {results['optimistic']['dice_mean']:.4f} | F1 instance {results['honest']['instance_f1']:.3f}")
    return results


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _write_threshold_curves(path, grid):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "threshold", "postprocess", "mean_dice"])
        for (mode, t, post), v in sorted(grid.items()):
            w.writerow([mode, t, post, round(float(v.mean()), 4)])


def _write_gallery(out_dir, eval_set, probs, stds, rows, min_area, n):
    """Les n pires et la meilleure image : image | probas | incertitude | erreurs."""
    order = np.argsort([r["dice"] for r in rows])
    picks = list(order[:n]) + ([order[-1]] if len(order) > n else [])
    lines = []
    for i in picks:
        r = rows[i]
        prob = probs[r["mode"]][i].astype(np.float32)
        std = stds[r["mode"]][i].astype(np.float32)
        pred = binarize(prob, r["threshold"], r["postprocess"], min_area)
        line = gallery_row(eval_set.images[i], eval_set.masks[i], pred, prob, std,
                           f"Dice {r['dice']:.3f} | {r['tp']}/{r['n_gt']} detectes, {r['fp']} FP")
        lines.append(cv2.resize(line, (line.shape[1] // 2, line.shape[0] // 2), interpolation=cv2.INTER_AREA))
    if lines:
        cv2.imwrite(str(out_dir / "error_gallery.jpg"), cv2.cvtColor(np.vstack(lines), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def _fmt_ci(ci):
    return f"[{ci[0]:.3f} ; {ci[1]:.3f}]"


def render_report(res: dict, rows: list[dict]) -> str:
    h, o = res["honest"], res["optimistic"]
    lines = [
        f"# Rapport d'évaluation : {res['name']}",
        "",
        f"{h['n_images']} images, {h['n_gt']} glomérules annotés. Réglages choisis par cross-fitting "
        f"({len(res['folds'])} plis) : chaque image est prédite avec des réglages choisis sans elle.",
        "",
        "| Protocole | Dice moyen | écart-type | IC 95 % (bootstrap) | F1 instance | Précision | Rappel |",
        "|---|---|---|---|---|---|---|",
        f"| **Honnête (cross-fitting)** | **{h['dice_mean']:.3f}** | {h['dice_std']:.3f} | {_fmt_ci(h['dice_ci95'])} "
        f"| {h['instance_f1']:.3f} | {h['instance_precision']:.3f} | {h['instance_recall']:.3f} |",
        f"| Optimiste (réglé sur le test) | {o['dice_mean']:.3f} | {o['dice_std']:.3f} | {_fmt_ci(o['dice_ci95'])} "
        f"| {o['instance_f1']:.3f} | {o['instance_precision']:.3f} | {o['instance_recall']:.3f} |",
        "",
        "## Modes de prédiction (seuil réglé par cross-fitting)",
        "",
        "| Mode | Dice | IC 95 % | F1 instance | s / image (CPU) | zoom auto moyen |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in res["by_mode"].items():
        lines.append(f"| {mode} | {m['dice_mean']:.3f} | {_fmt_ci(m['dice_ci95'])} | {m['instance_f1']:.3f} "
                     f"| {m['seconds_per_image']:.2f} | {m['mean_auto_zoom']:.2f} |")
    lines += [
        "",
        "## Erreurs (protocole honnête)",
        "",
        f"- Glomérules manqués (quasiment pas détectés) : {h['fn_missed']}",
        f"- Glomérules mal délimités (détectés mais IoU < seuil) : {h['fn_poorly_segmented']}",
        f"- Glomérules fusionnés avec un voisin : {h['merged']}",
        f"- Fausses détections (aucun glomérule dessous) : {h['fp_spurious']}",
        f"- Erreur de comptage moyenne : {h['count_mae']:.2f} glomérule(s) par image",
        "",
        "## Par image",
        "",
        "| Image | Zone | Dice | Réels | Prédits | VP | FP | FN | Mode | Seuil |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["dice"]):
        lines.append(f"| {r['image'][-40:]} | {r['group']} | {r['dice']:.3f} | {r['n_gt']} | {r['n_pred']} "
                     f"| {r['tp']} | {r['fp']} | {r['fn']} | {r['mode']} | {r['threshold']:.2f} |")
    d = res["deploy_settings"]
    lines += ["", f"Réglage de déploiement (choisi sur toutes les images, à n'utiliser que sur de nouvelles "
                  f"données) : mode `{d['mode']}`, seuil {d['threshold']:.2f}, post-traitement "
                  f"{'oui' if d['postprocess'] else 'non'}.", ""]
    return "\n".join(lines)
