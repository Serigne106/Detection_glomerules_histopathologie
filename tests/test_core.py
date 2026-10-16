import numpy as np
import pytest
import torch

from glomseg.augment import Augmenter
from glomseg.config import DEFAULTS, load_config, parse_overrides
from glomseg.data import make_folds, median_diameter, to_working_scale
from glomseg.inference import Predictor
from glomseg.io import rle_decode, rle_encode
from glomseg.metrics import bootstrap_ci, dice, instance_metrics, summarize
from glomseg.models import UNet, count_parameters
from glomseg.postprocess import clean_mask, label_instances
from glomseg.quantify import instances_table


def disk(shape, cx, cy, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


# ------------------------------------------------------------------ RLE / Dice
def test_rle_roundtrip_and_known_value():
    rng = np.random.default_rng(0)
    m = rng.random((37, 53)) > 0.7
    assert np.array_equal(rle_decode(rle_encode(m), m.shape), m)
    assert rle_encode(np.array([[0, 0, 1, 1, 1, 0, 1]])) == "3 3 7 1"
    assert rle_encode(np.zeros((4, 4))) == ""


def test_dice_cases():
    a = disk((64, 64), 32, 32, 10)
    assert dice(a, a) == 1.0
    assert dice(np.zeros((8, 8)), np.zeros((8, 8))) == 1.0
    assert dice(a, ~a) == 0.0
    with pytest.raises(ValueError):
        dice(a, a[:-1])


# ------------------------------------------------------------ instances
def test_instance_metrics_matching_and_error_types():
    shape = (200, 200)
    gt = np.zeros(shape, np.int32)
    gt[disk(shape, 40, 40, 15)] = 1          # bien détecté
    gt[disk(shape, 120, 40, 15)] = 2         # manqué
    gt[disk(shape, 60, 140, 15)] = 3         # fusionné avec 4
    gt[disk(shape, 88, 140, 15)] = 4
    pred = np.zeros(shape, np.int32)
    pred[disk(shape, 41, 40, 15)] = 1
    pred[disk(shape, 60, 140, 16) | disk(shape, 88, 140, 16)] = 2
    pred[(np.abs(np.mgrid[:200, :200][0] - 74) < 4) & (np.abs(np.mgrid[:200, :200][1] - 140) < 20)] = 2
    pred[disk(shape, 170, 170, 10)] = 3      # fausse détection
    m = instance_metrics(gt, pred, 0.5)
    assert (m["n_gt"], m["n_pred"], m["tp"]) == (4, 3, 1)
    assert m["fn"] == 3 and m["fp"] == 2
    assert m["fn_missed"] == 1
    assert m["merged"] == 2
    assert m["fp_spurious"] == 1
    assert m["count_error"] == -1
    assert 0 < m["f1"] < 1


def test_instance_metrics_empty():
    z = np.zeros((10, 10), np.int32)
    m = instance_metrics(z, z)
    assert m["f1"] == 1.0 or (m["tp"] == 0 and m["n_gt"] == 0)


def test_summary_and_bootstrap():
    rows = [dict(dice=d, tp=2, fp=1, fn=0, n_gt=2, n_pred=3, count_error=1, fn_missed=0,
                 fn_poorly_segmented=0, fp_spurious=1, merged=0) for d in (0.5, 0.7, 0.9)]
    s = summarize(rows, n_boot=200)
    assert abs(s["dice_mean"] - 0.7) < 1e-9
    lo, hi = s["dice_ci95"]
    assert 0.5 <= lo <= 0.7 <= hi <= 0.9
    assert abs(s["instance_precision"] - 2 / 3) < 1e-9
    assert bootstrap_ci([{"dice": 1}], lambda r: 1.0) == (1.0, 1.0)


# ------------------------------------------------------------- post-traitement
def test_clean_mask_removes_small_and_fills_holes():
    m = disk((100, 100), 50, 50, 20)
    m[48:52, 48:52] = False                   # trou
    m[5:8, 5:8] = True                        # tache
    out = clean_mask(m, min_area=50)
    assert out[50, 50] and not out[6, 6]


def test_split_touching_separates_two_disks():
    shape = (100, 140)
    m = disk(shape, 45, 50, 25) | disk(shape, 92, 50, 25)
    assert label_instances(m).max() == 1
    assert label_instances(m, split_touching=True).max() == 2
    assert label_instances(disk(shape, 50, 50, 20), split_touching=True).max() == 1


# -------------------------------------------------------------- données
def test_folds_cover_each_image_once():
    for groups in (["a", "a", "b", "b", "b"], ["x"] * 7):
        folds = make_folds(groups)
        tested = sorted(i for _, test in folds for i in test)
        assert tested == list(range(len(groups)))
        for tune, test in folds:
            assert not set(tune) & set(test) and tune


def test_augmenter_keeps_mask_aligned_and_binary():
    rng = np.random.default_rng(0)
    image = np.full((120, 120, 3), 200, np.uint8)
    mask = disk((120, 120), 60, 60, 20)
    image[mask] = (40, 40, 40)
    aug = Augmenter(dict(DEFAULTS["augment"], color=False, quality=False), crop_size=64)
    for _ in range(20):
        im, m = aug(image, mask, np.argwhere(mask), rng)
        assert im.shape == (64, 64, 3) and m.shape == (64, 64)
        assert set(np.unique(m)) <= {0, 1}
        if m.sum() > 50:                      # les pixels du masque restent sombres
            assert im[m.astype(bool)].mean() < im[~m.astype(bool)].mean()


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("name: a\ntrain:\n  epochs: 3\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nname: b\nmodel:\n  depth: 4\n")
    cfg = load_config(tmp_path / "child.yaml", parse_overrides(["train.lr=0.1"]))
    assert cfg["name"] == "b" and cfg["train"]["epochs"] == 3 and cfg["model"]["depth"] == 4
    assert cfg["train"]["lr"] == 0.1 and cfg["train"]["batch_size"] == DEFAULTS["train"]["batch_size"]


def test_median_diameter():
    inst = np.zeros((100, 100), np.int32)
    inst[disk((100, 100), 50, 50, 10)] = 1
    assert abs(median_diameter([inst]) - 20) < 1.5


# --------------------------------------------------------------- modèle / inférence
def test_unet_any_size():
    net = UNet(4, 3).eval()
    assert 10_000 < count_parameters(UNet()) / 1 and count_parameters(UNet()) > 1e6
    with torch.no_grad():
        for shape in [(1, 3, 64, 64), (1, 3, 50, 37)]:
            assert net(torch.zeros(shape)).shape == (1, 1, *shape[2:])


def test_predictor_shapes_tta_and_std():
    torch.manual_seed(0)
    p = Predictor(UNet(4, 3), scale=0.5)
    image = (np.random.default_rng(0).random((101, 87, 3)) * 255).astype(np.uint8)
    prob = p.predict(image)
    assert prob.shape == (101, 87) and 0 <= prob.min() and prob.max() <= 1
    mean, std = p.predict(image, tta=True, zooms=(0.8, 1.0), return_std=True)
    assert mean.shape == std.shape == (101, 87) and std.min() >= 0


def test_tiled_inference_matches_full_image():
    """La fusion des tuiles doit redonner (presque) la prédiction d'un seul passage."""
    torch.manual_seed(0)
    net = UNet(4, 3).eval()
    image = (np.random.default_rng(1).random((300, 260, 3)) * 255).astype(np.uint8)
    full = Predictor(net, scale=1.0, tile_size=10_000).predict(image)
    tiled = Predictor(net, scale=1.0, tile_size=128, tile_overlap=48).predict(image)
    assert np.abs(full - tiled)[32:-32, 32:-32].mean() < 0.02


def test_estimate_zoom_without_target_is_identity():
    p = Predictor(UNet(4, 3), scale=0.5)
    assert p.estimate_zoom(np.zeros((64, 64, 3), np.uint8)) == 1.0


def test_to_working_scale():
    im, m = to_working_scale(np.zeros((100, 60, 3), np.uint8), np.ones((100, 60), bool), 0.5)
    assert im.shape == (50, 30, 3) and m.shape == (50, 30) and m.all()


def test_instances_table():
    labels = np.zeros((50, 50), np.int32)
    labels[disk((50, 50), 20, 20, 8)] = 1
    std = np.zeros((50, 50), np.float32)
    std[labels == 1] = 0.3
    rows = instances_table(labels, np.ones((50, 50)), std, mpp=0.5)
    assert len(rows) == 1 and rows[0]["uncertain"] and abs(rows[0]["area_um2"] - rows[0]["area_px"] * 0.25) < 0.1


# ------------------------------------------------------------ contrôle des données
def test_check_data_flags_mask_on_background(tmp_path):
    import sys

    import cv2
    sys.path.insert(0, "scripts")
    from check_data import check_pair
    image = np.full((300, 300, 3), 245, np.uint8)            # verre blanc
    image[:, :150] = (200, 120, 170)                         # tissu rose à gauche
    good = np.zeros((300, 300), np.uint8)
    good[disk((300, 300), 70, 150, 30)] = 1                  # glomérule dans le tissu
    bad = np.zeros((300, 300), np.uint8)
    bad[disk((300, 300), 230, 150, 30)] = 1                  # "glomérule" sur le verre
    cv2.imwrite(str(tmp_path / "a.png"), image)
    for name, m in (("good", good), ("bad", bad)):
        cv2.imwrite(str(tmp_path / f"{name}_mask.tif"), m)
    assert check_pair(tmp_path / "a.png", tmp_path / "good_mask.tif")[2]["worst_object_on_background"] < 0.1
    assert check_pair(tmp_path / "a.png", tmp_path / "bad_mask.tif")[2]["worst_object_on_background"] > 0.9
