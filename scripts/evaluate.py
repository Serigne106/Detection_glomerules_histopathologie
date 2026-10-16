"""Évaluation honnête (cross-fitting) d'un ou plusieurs checkpoints.

    python scripts/evaluate.py --checkpoint runs/unet_scratch_seed42/model.pth
    python scripts/evaluate.py --checkpoint runs/unet_scratch_seed*/model.pth   # plusieurs seeds

Écrit pour chaque checkpoint : reports/<run>/{metrics.json, report.md, per_image.csv,
threshold_curves.csv, error_gallery.jpg} et ``model_deploy.pth`` (poids + réglage de déploiement).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from glomseg.checkpoint import load_predictor, save_checkpoint
from glomseg.config import deep_merge, load_config, parse_overrides
from glomseg.data import EvalSet, collect_pairs
from glomseg.evaluation import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--config", default=None, help="section eval/data à utiliser (défaut : celle du checkpoint)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    eval_set, summaries = None, []
    for ckpt_path in args.checkpoint:
        predictor, _, ckpt = load_predictor(ckpt_path, args.device)
        cfg = deep_merge(ckpt["config"], load_config(args.config) if args.config else {})
        cfg = deep_merge(cfg, parse_overrides(args.set))
        if args.data_root:
            cfg["data"]["root"] = args.data_root
        if args.eval_dir:
            cfg["data"]["eval_dir"] = args.eval_dir
        d = cfg["data"]
        if eval_set is None:
            pairs = collect_pairs(d["root"], d["eval_dir"], d["image_suffix"], d["mask_suffix"])
            if not pairs:
                raise SystemExit(f"Aucune paire trouvée dans {d['root']}/{d['eval_dir']}")
            eval_set = EvalSet(pairs, d["group_regex"])
            leak = set(ckpt["meta"].get("train_images", [])) & {p.name for p, _ in pairs}
            if leak:
                raise SystemExit(f"Fuite : {len(leak)} images d'évaluation ont servi à l'entraînement")
        run = Path(ckpt_path).parent.name
        print(f"\n=== {run} ===")
        res = evaluate(predictor, eval_set, cfg, Path(args.out) / run)
        res["run"] = run
        summaries.append(res)
        meta = {**ckpt["meta"], "inference": res["deploy_settings"]}
        save_checkpoint(Path(ckpt_path).with_name("model_deploy.pth"), predictor.model, ckpt["config"],
                        ckpt["history"], **meta)

    if len(summaries) > 1:
        d = np.array([s["honest"]["dice_mean"] for s in summaries])
        f = np.array([s["honest"]["instance_f1"] for s in summaries])
        agg = {"runs": [s["run"] for s in summaries],
               "dice_mean": float(d.mean()), "dice_std_seeds": float(d.std(ddof=1)),
               "f1_mean": float(f.mean()), "f1_std_seeds": float(f.std(ddof=1))}
        print(f"\nSur {len(d)} runs : Dice {agg['dice_mean']:.4f} ± {agg['dice_std_seeds']:.4f} (entre seeds), "
              f"F1 {agg['f1_mean']:.3f} ± {agg['f1_std_seeds']:.3f}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out) / f"seeds_{summaries[0]['name']}.json", "w") as fh:
            json.dump(agg, fh, indent=2)


if __name__ == "__main__":
    main()
