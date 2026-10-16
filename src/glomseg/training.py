"""Boucle d'entraînement (BCE + Dice, AdamW, OneCycle, EMA des poids)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import save_config
from .data import AugmentedDataset, EvalSet, median_diameter, worker_init_fn
from .inference import Predictor
from .metrics import dice
from .models import build_model, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (prob.sum() + target.sum() + smooth)


class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.bce(logits, target) + dice_loss(logits, target)


def _mlflow(cfg):
    if not cfg["train"].get("mlflow"):
        return None
    try:
        import mlflow
    except ImportError:
        print("mlflow non installé : suivi désactivé")
        return None
    mlflow.set_experiment("glomeruli-seg")
    mlflow.start_run(run_name=cfg["name"])
    flat = {f"{s}.{k}": v for s in ("input", "augment", "model", "train") for k, v in cfg[s].items()}
    mlflow.log_params({k: str(v) for k, v in flat.items()})
    return mlflow


def train(cfg: dict, train_pairs, out_dir, val_set: EvalSet | None = None, device="cpu", log=print) -> Path:
    """Entraîne et écrit ``out_dir/model.pth`` (+ ``model_best.pth`` si validation, ``history.json``).

    Par défaut (``train.checkpoint: last``) le modèle livré est celui de la DERNIÈRE epoch :
    le domaine d'évaluation ne sert qu'à tracer une courbe, jamais à choisir le modèle.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    tc = cfg["train"]
    seed_everything(tc["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))

    dataset = AugmentedDataset(train_pairs, cfg, seed=tc["seed"])
    loader = DataLoader(dataset, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                        num_workers=tc["num_workers"], worker_init_fn=worker_init_fn,
                        generator=torch.Generator().manual_seed(tc["seed"]))
    # diamètre médian des glomérules d'entraînement à la résolution de travail -> échelle automatique
    target_diameter = median_diameter([m.astype(np.int32) for m in _instances(train_pairs)], cfg["input"]["scale"])

    model = build_model(cfg["model"]).to(device)
    log(f"{cfg['name']} : {count_parameters(model) / 1e6:.2f} M paramètres, {len(train_pairs)} images, "
        f"{len(dataset)} crops augmentés / epoch, diamètre cible {target_diameter:.1f} px")
    criterion = BceDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=tc["lr"],
                                                    total_steps=tc["epochs"] * len(loader), pct_start=0.1)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tc["ema_decay"]), use_buffers=True)
    mlflow = _mlflow(cfg)

    history = {"epoch": [], "train_loss": [], "val_dice": [], "seconds": []}
    best = -1.0
    meta = {"target_diameter": float(target_diameter), "train_images": [p.name for p, _ in train_pairs]}
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)
            running += loss.item()
        train_loss = running / len(loader)
        val = float("nan")
        if val_set is not None and tc["val_every"] and (epoch % tc["val_every"] == 0 or epoch == tc["epochs"]):
            pred = Predictor(ema.module, cfg["input"]["scale"], cfg["input"]["normalization"], device)
            val = float(np.mean([dice(pred.predict(im) > 0.5, m) for im, m in zip(val_set.images, val_set.masks)]))
            if val > best:
                best = val
                save_checkpoint(out_dir / "model_best.pth", ema.module, cfg, history, selected_epoch=epoch, **meta)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_dice"].append(val)
        history["seconds"].append(round(time.time() - t0, 1))
        if mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            if np.isfinite(val):
                mlflow.log_metric("val_dice", val, step=epoch)
        log(f"epoch {epoch:3d}/{tc['epochs']}  loss={train_loss:.4f}  "
            f"val_dice={'—' if not np.isfinite(val) else f'{val:.4f}'}  {time.time() - t0:.0f}s")
        ema.module.train(False)

    save_checkpoint(out_dir / "model.pth", ema.module, cfg, history, selected_epoch=tc["epochs"], **meta)
    if tc["checkpoint"] == "best" and (out_dir / "model_best.pth").exists():
        log("ATTENTION : checkpoint 'best' choisi sur le domaine d'évaluation -> score optimiste.")
        (out_dir / "model.pth").unlink()
        (out_dir / "model_best.pth").rename(out_dir / "model.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    if mlflow:
        mlflow.log_artifact(str(out_dir / "model.pth"))
        mlflow.end_run()
    return out_dir / "model.pth"


def _instances(pairs):
    from .io import read_instances
    return [read_instances(m) for _, m in pairs]
