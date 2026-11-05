"""Export the separation models to ONNX (fp32 + fp16) and verify every file.

Models
  htdemucs_6s  - Demucs v4 hybrid transformer, 6 music stems (Meta AI, MIT)
  bandit_dnr   - BandIt Plus, speech / music / effects (ZFTurbo MSST, MIT)

Usage
  python tools/export_onnx.py --out dist/models \
      --bandit-config cfg.yaml --bandit-ckpt model.chpt      # real weights (CI)
  python tools/export_onnx.py --out /tmp/m --random-weights  # offline test

Writes <out>/<model>.onnx, <out>/<model>.fp16.onnx and <out>/models.json
"""
import argparse
import copy
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

from onnx_core import HTDemucsCore, half_positional_embeddings  # noqa: E402
from bandit_model import BanditPlus, load_bandit  # noqa: E402
from bunnys_engine import dsp  # noqa: E402

SOURCES_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
BANDIT_DEFAULT = dict(stems=["speech", "music", "effects"], n_bands=64, fs=44100, n_fft=2048,
                      in_channel=1, n_sqm_modules=8, emb_dim=128, rnn_dim=256, rnn_type="GRU",
                      mlp_dim=512)
BANDIT_CHUNK = 264600  # 6 s, BandIt Plus training chunk size

# The MHA "fast path" is an inference-only kernel with no ONNX mapping.
torch.backends.mha.set_fastpath_enabled(False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _export(module, args, path, input_names, output_names, opset):
    dyn = {n: {0: "batch"} for n in input_names + output_names}
    with torch.no_grad():
        torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dyn, opset_version=opset, do_constant_folding=True,
                          dynamo=False)


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _test_signal(n, sr, batch=2, seed=0):
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    sig = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)
           + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return np.stack([np.stack([sig, np.roll(sig, 100)])] * batch)  # [B, 2, n]


def _check(name, out, ref, tol):
    rel = float(np.abs(out - ref).max() / max(float(np.abs(ref).max()), 1e-8))
    print(f"[verify] {name}: rel err {rel:.3e} (tol {tol})")
    if not np.isfinite(out).all() or rel > tol:
        raise SystemExit(f"verification failed for {name}")
    return rel


# ---------------------------------------------------------------------------
# HTDemucs
# ---------------------------------------------------------------------------
def load_htdemucs(random_weights):
    if random_weights:
        from demucs.htdemucs import HTDemucs
        torch.manual_seed(0)
        model = HTDemucs(sources=SOURCES_6, samplerate=44100, segment=Fraction(39, 5))
    else:
        from demucs.pretrained import get_model
        model = get_model("htdemucs_6s").models[0]
    return model.eval()


def export_htdemucs(out, random_weights, opset):
    model = load_htdemucs(random_weights)
    L = int(model.segment * model.samplerate)
    mix = torch.randn(1, 2, L)
    mag = torch.from_numpy(dsp.model_spec(mix.numpy()))
    files = {"fp32": os.path.join(out, "htdemucs_6s.onnx"),
             "fp16": os.path.join(out, "htdemucs_6s.fp16.onnx")}
    _export(HTDemucsCore(model).eval(), (mix, mag), files["fp32"],
            ["mix", "mag"], ["spec_out", "time_out"], opset)
    with half_positional_embeddings():
        _export(HTDemucsCore(copy.deepcopy(model).half(), half=True).eval(), (mix, mag),
                files["fp16"], ["mix", "mag"], ["spec_out", "time_out"], opset)

    batch = _test_signal(L, model.samplerate)
    with torch.no_grad():
        ref = model(torch.from_numpy(batch)).numpy()
    m = dsp.model_spec(batch)
    for variant, tol in (("fp32", 1e-3), ("fp16", 2e-2)):
        so, to = _session(files[variant]).run(None, {"mix": batch, "mag": m})
        _check(f"htdemucs_6s {variant}", dsp.model_ispec(so, L) + to, ref, tol)
    return "htdemucs_6s", {
        "kind": "htdemucs", "sources": list(model.sources), "samplerate": model.samplerate,
        "segment_samples": L, "normalize": True,
    }, files


# ---------------------------------------------------------------------------
# BandIt Plus
# ---------------------------------------------------------------------------
def export_bandit(out, random_weights, config_path, ckpt_path, opset):
    if random_weights:
        torch.manual_seed(0)
        model = BanditPlus(**BANDIT_DEFAULT).eval()
    else:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model = load_bandit(cfg["model"], ckpt_path)
    L = BANDIT_CHUNK
    spec = torch.from_numpy(dsp.bandit_spec(np.zeros((1, 2, L), np.float32) + 0.01))
    files = {"fp32": os.path.join(out, "bandit_dnr.onnx"),
             "fp16": os.path.join(out, "bandit_dnr.fp16.onnx")}
    _export(model, (spec,), files["fp32"], ["spec"], ["stems"], opset)

    class Half(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m.half()

        def forward(self, s):
            return self.m(s.half()).float()

    _export(Half(copy.deepcopy(model)).eval(), (spec,), files["fp16"], ["spec"], ["stems"], opset)

    batch = _test_signal(L, 44100)
    sp = dsp.bandit_spec(batch)
    with torch.no_grad():
        ref = model(torch.from_numpy(sp)).numpy()
    ref_wave = dsp.bandit_ispec(ref, 2, 2, L)
    for variant, tol in (("fp32", 1e-3), ("fp16", 3e-2)):
        so = _session(files[variant]).run(None, {"spec": sp})[0]
        _check(f"bandit_dnr {variant}", dsp.bandit_ispec(so, 2, 2, L), ref_wave, tol)
    return "bandit_dnr", {
        "kind": "bandit", "sources": list(model.stems), "samplerate": 44100,
        "segment_samples": L, "normalize": False,
    }, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--bandit-config")
    ap.add_argument("--bandit-ckpt")
    ap.add_argument("--only", choices=["htdemucs_6s", "bandit_dnr"])
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = {"version": 2, "models": {}}
    jobs = []
    if args.only in (None, "htdemucs_6s"):
        jobs.append(lambda: export_htdemucs(args.out, args.random_weights, args.opset))
    if args.only in (None, "bandit_dnr"):
        jobs.append(lambda: export_bandit(args.out, args.random_weights, args.bandit_config,
                                          args.bandit_ckpt, args.opset))
    for job in jobs:
        name, info, files = job()
        info["variants"] = {k: {"file": os.path.basename(p), "sha256": sha256(p),
                                "size": os.path.getsize(p)} for k, p in files.items()}
        manifest["models"][name] = info

    with open(os.path.join(args.out, "models.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
