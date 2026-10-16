import base64

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from glomseg.checkpoint import save_checkpoint  # noqa: E402
from glomseg.config import load_config  # noqa: E402
from glomseg.models import UNet  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = load_config(overrides={"model": {"base_channels": 4, "depth": 3}})
    path = tmp_path / "model.pth"
    save_checkpoint(path, UNet(4, 3), cfg, target_diameter=40.0,
                    inference={"mode": "simple", "threshold": 0.5, "postprocess": False, "min_area": 10})
    monkeypatch.setenv("MODEL_PATH", str(path))
    from app import api
    api.get_model.cache_clear()
    return TestClient(api.app)


def png_bytes(h=120, w=90):
    img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["settings"]["mode"] == "simple"


def test_predict(client):
    r = client.post("/predict?overlay=true&rle=true&mode=tta8&threshold=0.3",
                    files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["image_size"] == [90, 120] and js["settings"]["mode"] == "tta8"
    assert js["summary"]["n_glomeruli"] == len(js["glomeruli"])
    overlay = cv2.imdecode(np.frombuffer(base64.b64decode(js["overlay_png_base64"]), np.uint8), cv2.IMREAD_COLOR)
    assert overlay.shape == (120, 90, 3)


def test_predict_errors(client):
    assert client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400
    assert client.post("/predict?mode=foo", files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 422
