"""Bout en bout sur données synthétiques : entraînement -> évaluation -> checkpoint -> ONNX."""
import json

import pytest

from glomseg.checkpoint import load_predictor, save_checkpoint
from glomseg.data import EvalSet, collect_pairs
from glomseg.evaluation import evaluate
from glomseg.inference import full_pipeline
from glomseg.training import train


@pytest.fixture
def trained(tiny_cfg, tmp_path):
    d = tiny_cfg["data"]
    pairs = collect_pairs(d["root"], d["train_dirs"])
    val = EvalSet(collect_pairs(d["root"], d["eval_dir"]), d["group_regex"])
    path = train(tiny_cfg, pairs, tmp_path / "run", val_set=val, log=lambda *a: None)
    return path, val


def test_train_writes_checkpoint_and_history(trained):
    path, _ = trained
    assert path.exists()
    hist = json.loads((path.parent / "history.json").read_text())
    assert len(hist["train_loss"]) == 2
    predictor, settings, ckpt = load_predictor(path)
    assert settings is None and ckpt["meta"]["target_diameter"] > 0


def test_evaluate_outputs(trained, tiny_cfg, tmp_path):
    path, val = trained
    predictor, _, _ = load_predictor(path)
    res = evaluate(predictor, val, tiny_cfg, tmp_path / "report", log=lambda *a: None)
    for f in ("metrics.json", "report.md", "per_image.csv", "threshold_curves.csv", "error_gallery.jpg"):
        assert (tmp_path / "report" / f).exists(), f
    assert len(res["folds"]) == 2                      # 2 zones de lame dans les noms synthétiques
    assert res["optimistic"]["dice_mean"] >= res["honest"]["dice_mean"] - 1e-9
    assert set(res["by_mode"]) == {"simple", "tta8", "auto"}


def test_full_pipeline_and_deploy_checkpoint(trained, tmp_path):
    path, val = trained
    predictor, _, ckpt = load_predictor(path)
    settings = {"mode": "tta8", "threshold": 0.5, "postprocess": True, "min_area": 50}
    save_checkpoint(tmp_path / "deploy.pth", predictor.model, ckpt["config"], ckpt["history"],
                    **ckpt["meta"], inference=settings)
    p2, s2, _ = load_predictor(tmp_path / "deploy.pth")
    assert s2 == settings
    prob, std, mask, labels, info = full_pipeline(p2, val.images[0], s2)
    assert prob.shape == mask.shape == labels.shape == val.images[0].shape[:2]


def test_onnx_export_matches_torch(trained, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import sys
    sys.path.insert(0, "scripts")
    from export_onnx import check, export
    path, _ = trained
    onnx_path, model = export(path, tmp_path / "m.onnx")
    assert check(onnx_path, model, shape=(1, 3, 96, 64)) < 1e-3
