"""Contrôle qualité des données avant entraînement.

Pour chaque paire image / masque : nombre et taille des glomérules, et part du masque posée sur
du FOND (verre blanc, sans tissu). Une part élevée trahit un masque mal apparié, décalé ou mal
annoté : ce contrôle a détecté une inversion de deux tuiles pendant le développement du projet.

    python scripts/check_data.py --dir Set02_Glom_train_clean/train03 --out reports/data_check
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from glomseg.io import list_pairs, read_instances, read_rgb


def background_mask(image, sat_max=25, val_min=200, kernel=41, min_area_frac=0.02):
    """Fond = grandes zones très claires et peu saturées (verre de la lame).

    L'ouverture morphologique à gros noyau et le filtre de surface écartent les zones pâles
    DU tissu (espace de Bowman, lumière des tubules), qui sont petites ou fines.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pale = ((hsv[..., 1] < sat_max) & (hsv[..., 2] > val_min)).astype(np.uint8)
    pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pale, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_frac * pale.size
    return keep[labels]


def check_pair(img_path, mask_path):
    image, inst = read_rgb(img_path), read_instances(mask_path)
    fg = inst > 0
    bg = background_mask(image)
    ids = [i for i in np.unique(inst) if i > 0]
    per_obj = [bg[inst == i].mean() for i in ids]
    return image, inst, {
        "image": img_path.name,
        "size": f"{image.shape[1]}x{image.shape[0]}",
        "shape_match": image.shape[:2] == inst.shape,
        "n_glomeruli": len(ids),
        "fg_percent": round(100 * fg.mean(), 2),
        "mask_on_background": round(float(bg[fg].mean()) if fg.any() else 0.0, 3),
        "worst_object_on_background": round(float(max(per_obj)) if per_obj else 0.0, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="reports/data_check")
    ap.add_argument("--max-background", type=float, default=0.5,
                    help="seuil d'alerte : part d'un glomérule annoté posée sur du fond")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, orphans = list_pairs(args.dir)
    rows, flagged = [], 0
    for img_path, mask_path in pairs:
        image, inst, row = check_pair(img_path, mask_path)
        row["alert"] = (not row["shape_match"]) or row["worst_object_on_background"] > args.max_background
        flagged += row["alert"]
        rows.append(row)
        if row["alert"]:
            vis = image.copy()
            contours, _ = cv2.findContours((inst > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, contours, -1, (0, 220, 0), 4)
            cv2.imwrite(str(out / f"ALERT_{img_path.stem}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"{'ALERTE' if row['alert'] else 'ok    '} {row['image']:<45} {row['n_glomeruli']:3d} glom. "
              f"| masque sur fond : {row['mask_on_background']:.2f} "
              f"(pire objet {row['worst_object_on_background']:.2f})")
    if rows:
        with open(out / "data_check.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(pairs)} paires, {len(orphans)} masques sans image, {flagged} alerte(s) -> {out}")


if __name__ == "__main__":
    main()
