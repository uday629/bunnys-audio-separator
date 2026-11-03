"""Export HTDemucs (6 stems) to ONNX (fp32) and verify it against PyTorch.

Usage:
    python tools/export_onnx.py --out dist/models               # real pretrained weights
    python tools/export_onnx.py --out /tmp/m --random-weights   # offline test

Writes <out>/htdemucs_6s.onnx and <out>/models.json
"""
import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))

from onnx_core import HTDemucsCore  # noqa: E402
from bunnys_engine import dsp  # noqa: E402

SOURCES_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]


def load_model(name, random_weights):
    if random_weights:
        from demucs.htdemucs import HTDemucs
        torch.manual_seed(0)
        model = HTDemucs(sources=SOURCES_6, samplerate=44100, segment=Fraction(39, 5))
    else:
        from demucs.pretrained import get_model
        bag = get_model(name)
        model = bag.models[0]
    model.eval()
    return model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(model, path, opset):
    core = HTDemucsCore(model).eval()
    L = int(model.segment * model.samplerate)
    mix = torch.randn(1, 2, L)
    mag = torch.from_numpy(dsp.model_spec(mix.numpy()))
    torch.onnx.export(
        core, (mix, mag), path,
        input_names=["mix", "mag"],
        output_names=["spec_out", "time_out"],
        dynamic_axes={"mix": {0: "batch"}, "mag": {0: "batch"},
                      "spec_out": {0: "batch"}, "time_out": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return L


def verify(model, onnx_path, L, tol):
    """Full-pipeline check: torch HTDemucs.forward vs NumPy-DSP + ONNX core."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    t = np.arange(L) / model.samplerate
    rng = np.random.default_rng(0)
    sig = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)
           + 0.05 * rng.standard_normal(L)).astype(np.float32)
    batch = np.stack([np.stack([sig, np.roll(sig, 100)])] * 2)  # B=2 checks dynamic batch
    with torch.no_grad():
        ref = model(torch.from_numpy(batch)).numpy()
    mag = dsp.model_spec(batch)
    spec_out, time_out = sess.run(None, {"mix": batch, "mag": mag})
    out = dsp.model_ispec(spec_out, L) + time_out
    err = float(np.abs(out - ref).max())
    scale = float(np.abs(ref).max())
    rel = err / max(scale, 1e-8)
    print(f"[verify] {os.path.basename(onnx_path)}: max abs err {err:.3e}, rel {rel:.3e}")
    if rel > tol:
        raise SystemExit(f"verification failed for {onnx_path}: rel err {rel} > {tol}")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="htdemucs_6s")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.name, args.random_weights)
    fp32 = os.path.join(args.out, f"{args.name}.onnx")
    L = export(model, fp32, args.opset)
    verify(model, fp32, L, tol=1e-3)

    # NOTE: an fp16 variant is planned; both onnxconverter-common and native
    # torch half export currently fail on the HTDemucs cross-transformer.
    variants = {"fp32": fp32}

    manifest = {
        "version": 1,
        "models": {
            args.name: {
                "sources": list(model.sources),
                "samplerate": model.samplerate,
                "segment_samples": L,
                "variants": {
                    k: {"file": os.path.basename(p), "sha256": sha256(p),
                        "size": os.path.getsize(p)}
                    for k, p in variants.items()
                },
            }
        },
    }
    with open(os.path.join(args.out, "models.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
