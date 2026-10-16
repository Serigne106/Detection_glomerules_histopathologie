"""Figures du README : courbes d'apprentissage, Dice selon le seuil, Dice par image.

    python scripts/make_figures.py --runs runs --reports reports --out docs/figures
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
PALETTE = ["#2f6fdb", "#e0782f", "#1f9e74", "#b8508f", "#8a6d1f", "#5b5fc7"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def training_curves(runs_dir, out):
    hists = {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/history.json"))}
    if not hists:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (name, h) in enumerate(hists.items()):
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(h["epoch"], h["train_loss"], color=c, lw=1.8, label=name)
        ep = [e for e, v in zip(h["epoch"], h["val_dice"]) if v == v]
        vd = [v for v in h["val_dice"] if v == v]
        if vd:
            axes[1].plot(ep, vd, color=c, lw=1.8, marker="o", ms=3, label=name)
    style(axes[0], "Perte d'entraînement (BCE + Dice)", "epoch", "perte")
    style(axes[1], "Dice sur le domaine jamais vu (suivi seulement)", "epoch", "Dice (seuil 0.5)")
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


def threshold_and_images(report_dir, out):
    curves = defaultdict(lambda: ([], []))
    with open(report_dir / "threshold_curves.csv") as f:
        for r in csv.DictReader(f):
            if r["postprocess"] == "True":
                curves[r["mode"]][0].append(float(r["threshold"]))
                curves[r["mode"]][1].append(float(r["mean_dice"]))
    with open(report_dir / "per_image.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["dice"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    for i, (mode, (t, d)) in enumerate(curves.items()):
        axes[0].plot(t, d, color=PALETTE[i % len(PALETTE)], lw=1.8, marker="o", ms=3, label=mode)
    style(axes[0], "Dice moyen selon le seuil (avec post-traitement)", "seuil", "Dice")
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    dices = [float(r["dice"]) for r in rows]
    labels = [f"{r['image'].split('png_')[-1][-18:]}  ({r['tp']}/{r['n_gt']})" for r in rows]
    axes[1].barh(range(len(rows)), dices, color=PALETTE[0], height=0.6)
    axes[1].set_yticks(range(len(rows)), labels, fontsize=8)
    axes[1].axvline(np.mean(dices), color=INK, lw=1, ls="--")
    axes[1].text(np.mean(dices) + 0.01, len(rows) - 0.6, f"moyenne {np.mean(dices):.3f}", color=INK, fontsize=8)
    style(axes[1], "Dice par image, protocole honnête (détectés / réels)", "Dice", "")
    axes[1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out / f"evaluation_{report_dir.name}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    training_curves(args.runs, out)
    for rep in sorted(Path(args.reports).glob("*/threshold_curves.csv")):
        threshold_and_images(rep.parent, out)
    print(f"Figures -> {out}")


if __name__ == "__main__":
    main()
