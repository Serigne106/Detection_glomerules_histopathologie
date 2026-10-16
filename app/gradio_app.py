"""Démo interactive (Hugging Face Spaces ou en local).

    MODEL_PATH=runs/unet_scratch_seed42/model_deploy.pth python app/gradio_app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from glomseg.checkpoint import load_predictor
from glomseg.inference import full_pipeline
from glomseg.io import to_rgb_uint8
from glomseg.quantify import instances_table, summary
from glomseg.viz import contour_overlay, heatmap

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_deploy.pth")
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", "examples"))
PREDICTOR, SETTINGS, _ = load_predictor(MODEL_PATH)
SETTINGS = dict(SETTINGS or {"mode": "tta8", "threshold": 0.5, "postprocess": True, "min_area": 1000})

MODE_LABELS = {
    "Rapide (1 passe)": "simple",
    "Robuste (8 symétries)": "tta8",
    "Échelle automatique (taille des glomérules estimée)": "auto",
}
COLUMNS = ["id", "area_px", "eq_diameter_px", "mean_prob", "mean_uncertainty", "uncertain"]


def run(image, mode_label, threshold, mpp):
    if image is None:
        return None, None, "Déposez une image.", None, None
    image = to_rgb_uint8(np.asarray(image))
    settings = dict(SETTINGS, mode=MODE_LABELS[mode_label], threshold=float(threshold))
    prob, std, mask, labels, info = full_pipeline(PREDICTOR, image, settings)
    table = instances_table(labels, prob, std, mpp or None)
    s = summary(table)
    text = (f"**{s['n_glomeruli']} glomérule(s) détecté(s)**"
            + (f", dont {s.get('n_uncertain', 0)} incertain(s) (contour jaune, à vérifier)" if table else "")
            + f"  \nZoom appliqué : ×{info['zoom']:.2f} · temps : {info['seconds']:.1f} s")
    if table:
        text += f"  \nDiamètre équivalent médian : {s['median_eq_diameter_px']:.0f} px"
    rows = [[r.get(c) for c in COLUMNS] for r in table]
    csv_path = None
    if table:
        csv_path = Path(tempfile.mkdtemp()) / "glomeruli.csv"
        keys = list(table[0].keys())
        csv_path.write_text(",".join(keys) + "\n" + "\n".join(",".join(str(r[k]) for k in keys) for r in table))
    return contour_overlay(image, labels, table), heatmap(std, 0.5), text, rows, csv_path


with gr.Blocks(title="Glomeruli finder") as demo:
    gr.Markdown(
        "# Détection de glomérules rénaux\n"
        "U-Net résiduel de 2 M de paramètres, **entraîné de zéro** sur 42 images de rein de rongeur "
        "et évalué sur du rein humain pathologique jamais vu. Tourne sur CPU.\n\n"
        "*Outil de recherche, pas un dispositif médical.*")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Image de tissu rénal (H&E, PAS, ...)")
            mode = gr.Radio(list(MODE_LABELS), value="Robuste (8 symétries)", label="Mode")
            thr = gr.Slider(0.2, 0.8, value=SETTINGS["threshold"], step=0.05, label="Seuil de décision")
            mpp = gr.Number(value=None, label="Microns par pixel (optionnel, pour les surfaces en µm²)")
            btn = gr.Button("Analyser", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Glomérules détectés (jaune = incertain)")
            out_unc = gr.Image(label="Carte d'incertitude (désaccord entre les passes)")
    out_txt = gr.Markdown()
    out_tab = gr.Dataframe(headers=COLUMNS, label="Quantification par glomérule")
    out_csv = gr.File(label="Télécharger le CSV")
    btn.click(run, [inp, mode, thr, mpp], [out_img, out_unc, out_txt, out_tab, out_csv])
    if EXAMPLES_DIR.exists():
        gr.Examples([[str(p)] for p in sorted(EXAMPLES_DIR.glob("*.png"))], inputs=[inp])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
