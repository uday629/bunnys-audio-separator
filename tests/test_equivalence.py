"""Check the ONNX engine against the reference PyTorch demucs pipeline.

    python tests/test_equivalence.py <models_dir> [--random-weights]

Uses models_dir/models.json + htdemucs_6s.onnx produced by tools/export_onnx.py.
With --random-weights the torch model is rebuilt with the same seed as the export.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bunnys_engine.separator import Separator  # noqa: E402
from export_onnx import load_htdemucs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models_dir")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    with open(os.path.join(args.models_dir, "models.json")) as f:
        info = json.load(f)["models"]["htdemucs_6s"]
    model = load_htdemucs(args.random_weights)
    sr = info["samplerate"]
    n = int(args.seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    wav = np.stack([
        0.3 * np.sin(2 * np.pi * 110 * t) * (1 + np.sin(2 * np.pi * 0.5 * t)),
        0.3 * np.sin(2 * np.pi * 165 * t) + 0.05 * rng.standard_normal(n),
    ]).astype(np.float32)

    path = os.path.join(args.models_dir, "htdemucs_6s.onnx")
    sep = Separator(info, lambda variant: path, device="cpu")
    ours = sep.separate(wav, "balanced")

    from demucs.apply import apply_model
    ref = wav.mean(0)
    mix = torch.from_numpy((wav - ref.mean()) / ref.std())[None]
    with torch.no_grad():
        theirs = apply_model(model, mix, shifts=0, split=True, overlap=0.25, progress=False)[0]
    theirs = theirs.numpy() * ref.std() + ref.mean()

    worst = 0.0
    for i, s in enumerate(info["sources"]):
        err = np.abs(ours[s] - theirs[i]).max()
        rel = err / max(np.abs(theirs[i]).max(), 1e-8)
        worst = max(worst, rel)
        print(f"{s:8s} max abs err {err:.2e}  rel {rel:.2e}")
    print(f"worst relative error: {worst:.2e}")
    if worst > 5e-3:
        raise SystemExit("MISMATCH between ONNX engine and PyTorch demucs")
    print("OK: ONNX engine matches PyTorch demucs")


if __name__ == "__main__":
    main()
