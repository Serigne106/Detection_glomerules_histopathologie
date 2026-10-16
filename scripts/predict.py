"""Prédiction sur un dossier : soumission RLE + tableau des glomérules + superpositions.

    python scripts/predict.py --checkpoint runs/.../model_deploy.pth --input dossier_images --out predictions/
"""
import argparse
import csv
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from glomseg.checkpoint import load_predictor
from glomseg.inference import full_pipeline
from glomseg.io import list_images, read_rgb, rle_encode
from glomseg.quantify import instances_table
from glomseg.viz import contour_overlay

DEFAULT_SETTINGS = {"mode": "tta8", "threshold": 0.5, "postprocess": True, "min_area": 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--mode", default=None, help="surcharge le mode enregistré (simple, tta8, auto, ...)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--mpp", type=float, default=None, help="microns par pixel (surfaces en µm²)")
    ap.add_argument("--split-touching", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    predictor, settings, _ = load_predictor(args.checkpoint, args.device)
    settings = dict(settings or DEFAULT_SETTINGS)
    if args.mode:
        settings["mode"] = args.mode
    if args.threshold is not None:
        settings["threshold"] = args.threshold
    print(f"Réglages : {settings}")

    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    paths = list_images(args.input)
    all_rows = []
    with open(out / "submission.csv", "w", newline="") as f:
        sub = csv.writer(f)
        sub.writerow(["ID", "mask"])
        for p in tqdm(paths, desc="Prédiction"):
            image = read_rgb(p)
            prob, std, mask, labels, info = full_pipeline(predictor, image, settings, args.split_touching)
            sub.writerow([p.stem, rle_encode(mask)])
            table = instances_table(labels, prob, std, args.mpp)
            all_rows += [{"image": p.stem, "zoom": round(info["zoom"], 3), **r} for r in table]
            if not args.no_overlays:
                cv2.imwrite(str(out / "overlays" / f"{p.stem}.jpg"),
                            cv2.cvtColor(contour_overlay(image, labels, table), cv2.COLOR_RGB2BGR))
    if all_rows:
        with open(out / "glomeruli.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"{len(paths)} images -> {out}/submission.csv, {len(all_rows)} glomérules -> {out}/glomeruli.csv")


if __name__ == "__main__":
    main()
