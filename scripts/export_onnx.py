"""Export ONNX (taille d'entrée dynamique) et vérification numérique contre PyTorch.

    python scripts/export_onnx.py --checkpoint runs/unet_scratch_seed42/model_deploy.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from glomseg.checkpoint import load_checkpoint


def export(ckpt_path, out_path=None, opset=17):
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    out_path = Path(out_path or Path(ckpt_path).with_suffix(".onnx"))
    dummy = torch.randn(1, 3, 384, 384)
    torch.onnx.export(model, dummy, str(out_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "logits": {0: "batch", 2: "height", 3: "width"}},
                      opset_version=opset, dynamo=False)
    return out_path, model


def check(onnx_path, model, shape=(1, 3, 384, 352)):
    import onnxruntime as ort
    x = torch.randn(*shape)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]
    return float(np.abs(out - ref).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path, model = export(args.checkpoint, args.out)
    err = check(path, model)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} Mo) - écart max PyTorch/ONNX = {err:.2e}")


if __name__ == "__main__":
    main()
