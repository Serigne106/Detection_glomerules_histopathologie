"""Entraînement.

    python scripts/train.py --config configs/unet_scratch.yaml --seed 42
    python scripts/train.py --config configs/unet_pretrained.yaml --set train.epochs=30
"""
import argparse
from pathlib import Path

import torch

from glomseg.config import load_config, parse_overrides
from glomseg.data import EvalSet, collect_pairs
from glomseg.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="dossier du run (défaut : runs/<name>_seed<seed>)")
    ap.add_argument("--set", nargs="*", default=[], help="surcharges, ex. train.epochs=5")
    ap.add_argument("--no-val", action="store_true", help="pas de courbe de validation pendant l'entraînement")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    overrides = parse_overrides(args.set)
    if args.seed is not None:
        overrides.setdefault("train", {})["seed"] = args.seed
    cfg = load_config(args.config, overrides)
    d = cfg["data"]
    train_pairs = collect_pairs(d["root"], d["train_dirs"], d["image_suffix"], d["mask_suffix"])
    if not train_pairs:
        raise SystemExit(f"Aucune paire image/masque trouvée dans {d['root']}/{d['train_dirs']}")
    val_set = None
    eval_dir = Path(d["root"]) / d["eval_dir"] if d.get("eval_dir") else None
    if not args.no_val and eval_dir and eval_dir.exists():
        val_set = EvalSet(collect_pairs(d["root"], d["eval_dir"], d["image_suffix"], d["mask_suffix"]),
                          d["group_regex"])
    out = args.out or f"runs/{cfg['name']}_seed{cfg['train']['seed']}"
    path = train(cfg, train_pairs, out, val_set=val_set, device=args.device)
    print(f"Modèle : {path}")


if __name__ == "__main__":
    main()
