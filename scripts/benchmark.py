"""Latence CPU par mode de prédiction (PyTorch) et du réseau seul (PyTorch vs ONNX Runtime).

    python scripts/benchmark.py --checkpoint runs/.../model_deploy.pth --image data/xxx.png
"""
import argparse
import json
import time

import numpy as np
import torch

from glomseg.checkpoint import load_predictor
from glomseg.inference import MODES
from glomseg.io import read_rgb


def timeit(fn, repeat=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None, help="image de test (défaut : bruit 1024x1024)")
    ap.add_argument("--onnx", default=None, help="modèle ONNX à comparer (optionnel)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    predictor, _, _ = load_predictor(args.checkpoint)
    image = read_rgb(args.image) if args.image else (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
    res = {"image_shape": list(image.shape), "threads": torch.get_num_threads(), "modes": {}}
    print(f"Image {image.shape[1]}x{image.shape[0]}, {res['threads']} threads CPU")
    for mode in MODES:
        s = timeit(lambda m=mode: predictor.predict_mode(image, m), repeat=args.repeat)
        res["modes"][mode] = s
        print(f"  {mode:<18} {s:6.2f} s/image")

    x = torch.randn(1, 3, round(image.shape[0] * predictor.scale), round(image.shape[1] * predictor.scale))
    with torch.no_grad():
        res["torch_forward"] = timeit(lambda: predictor.model(x), repeat=10)
    print(f"  réseau seul PyTorch    {res['torch_forward'] * 1000:6.1f} ms")
    if args.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        xn = x.numpy()
        res["onnx_forward"] = timeit(lambda: sess.run(None, {"image": xn}), repeat=10)
        print(f"  réseau seul ONNX       {res['onnx_forward'] * 1000:6.1f} ms "
              f"(x{res['torch_forward'] / res['onnx_forward']:.2f})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
