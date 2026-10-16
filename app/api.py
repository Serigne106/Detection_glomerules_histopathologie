"""API REST de prédiction.

    MODEL_PATH=runs/unet_scratch_seed42/model_deploy.pth uvicorn app.api:app --port 8000
    curl -F "file=@image.png" "localhost:8000/predict?overlay=true"
"""
from __future__ import annotations

import base64
import os
import time
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from glomseg import __version__
from glomseg.checkpoint import load_predictor
from glomseg.inference import MODES, full_pipeline
from glomseg.io import rle_encode, to_rgb_uint8
from glomseg.quantify import instances_table, summary
from glomseg.viz import contour_overlay

DEFAULT_SETTINGS = {"mode": "tta8", "threshold": 0.5, "postprocess": True, "min_area": 1000}
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 4096 * 4096))

app = FastAPI(title="Glomeruli segmentation API", version=__version__,
              description="Détection, segmentation et quantification de glomérules rénaux. "
                          "Outil de recherche : pas un dispositif médical.")


@lru_cache(maxsize=1)
def get_model():
    path = os.environ.get("MODEL_PATH", "models/model_deploy.pth")
    if not os.path.exists(path):
        raise HTTPException(503, f"Modèle introuvable : {path} (variable MODEL_PATH)")
    predictor, settings, ckpt = load_predictor(path)
    return predictor, dict(settings or DEFAULT_SETTINGS), ckpt["config"]["name"]


def decode_image(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(400, "Fichier image illisible")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise HTTPException(413, f"Image trop grande (> {MAX_PIXELS} pixels)")
    return to_rgb_uint8(image, bgr=True)


@app.get("/health")
def health():
    predictor, settings, name = get_model()
    return {"status": "ok", "model": name, "settings": settings, "version": __version__}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  mode: str | None = Query(None, description=f"un de {list(MODES)}"),
                  threshold: float | None = Query(None, ge=0.0, le=1.0),
                  mpp: float | None = Query(None, gt=0, description="microns par pixel"),
                  overlay: bool = Query(False, description="renvoyer l'image annotée (PNG base64)"),
                  rle: bool = Query(False, description="renvoyer le masque en RLE")):
    predictor, settings, name = get_model()
    settings = dict(settings)
    if mode is not None:
        if mode not in MODES:
            raise HTTPException(422, f"mode inconnu : {mode}")
        settings["mode"] = mode
    if threshold is not None:
        settings["threshold"] = threshold
    image = decode_image(await file.read())
    t0 = time.perf_counter()
    prob, std, mask, labels, info = full_pipeline(predictor, image, settings)
    table = instances_table(labels, prob, std, mpp)
    out = {
        "model": name,
        "settings": settings,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "zoom": round(info["zoom"], 3),
        "seconds": round(time.perf_counter() - t0, 3),
        "summary": summary(table),
        "glomeruli": table,
    }
    if rle:
        out["rle"] = rle_encode(mask)
    if overlay:
        ok, png = cv2.imencode(".png", cv2.cvtColor(contour_overlay(image, labels, table), cv2.COLOR_RGB2BGR))
        out["overlay_png_base64"] = base64.b64encode(png.tobytes()).decode()
    return out
