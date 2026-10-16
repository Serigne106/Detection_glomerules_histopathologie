"""Tableau comparatif de toutes les expériences (de zéro, pré-entraîné, ablations, seeds).

    python scripts/compare.py --reports reports --out reports/comparison.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = {
    "unet_scratch": "U-Net résiduel, de zéro (référence)",
    "unet_pretrained": "U-Net ResNet34 pré-entraîné ImageNet",
    "ablation_no_color": "Référence sans augmentation de couleur",
    "ablation_no_scale": "Référence sans zoom aléatoire",
    "ablation_no_quality": "Référence sans flou / bruit / JPEG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="reports/comparison.md")
    args = ap.parse_args()

    groups = defaultdict(list)
    for p in sorted(Path(args.reports).glob("*/metrics.json")):
        res = json.loads(p.read_text())
        groups[res["name"]].append(res)
    if not groups:
        raise SystemExit("Aucun metrics.json trouvé")

    def ms(values):
        v = np.array(values)
        return f"{v.mean():.3f}" + (f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else "")

    lines = ["| Expérience | runs | Dice (honnête) | F1 instance | Rappel | Précision | meilleur mode | s / image |",
             "|---|---|---|---|---|---|---|---|"]
    for name, runs in groups.items():
        h = [r["honest"] for r in runs]
        best_mode = max(runs[0]["by_mode"], key=lambda m: runs[0]["by_mode"][m]["dice_mean"])
        sec = np.mean([r["by_mode"][best_mode]["seconds_per_image"] for r in runs])
        lines.append(f"| {LABELS.get(name, name)} | {len(runs)} | {ms([x['dice_mean'] for x in h])} "
                     f"| {ms([x['instance_f1'] for x in h])} | {ms([x['instance_recall'] for x in h])} "
                     f"| {ms([x['instance_precision'] for x in h])} | {best_mode} | {sec:.2f} |")
    lines.append("\n± = écart-type entre seeds d'entraînement. Dice et F1 mesurés par cross-fitting "
                 "(réglages choisis sans l'image testée).")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
