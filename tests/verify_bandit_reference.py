"""CI-only: compare our BandIt ONNX pipeline with the ORIGINAL implementation.

    python tests/verify_bandit_reference.py <mss_repo_dir> <config.yaml> <ckpt> <models_dir>

<mss_repo_dir> is a checkout of ZFTurbo/Music-Source-Separation-Training.
Weights are loaded with weights_only=True (no pickle code execution).
"""
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "engine"))


def main():
    repo, cfg_path, ckpt, models_dir = sys.argv[1:5]
    sys.path.insert(0, repo)
    # `asteroid` is only used for training losses; stub it so we don't pull it in.
    import types
    stub = types.ModuleType("asteroid")
    stub.losses = types.ModuleType("asteroid.losses")
    sys.modules.setdefault("asteroid", stub)
    sys.modules.setdefault("asteroid.losses", stub.losses)
    from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
    from bunnys_engine import dsp
    import onnxruntime as ort

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ref_model = MultiMaskMultiSourceBandSplitRNNSimple(**cfg["model"])
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    ref_model.load_state_dict(state)
    ref_model.eval()

    L = cfg["audio"]["chunk_size"]
    rng = np.random.default_rng(3)
    t = np.arange(L) / 44100
    x = np.stack([
        0.2 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.05 * rng.standard_normal(L),
        0.2 * np.sin(2 * np.pi * 523 * t) + 0.05 * rng.standard_normal(L),
    ]).astype(np.float32)[None]
    with torch.no_grad():
        ref = ref_model(torch.from_numpy(x)).numpy()  # [1, S, 2, L]

    for variant, fname, tol in (("fp32", "bandit_dnr.onnx", 2e-3), ("fp16", "bandit_dnr.fp16.onnx", 5e-2)):
        sess = ort.InferenceSession(os.path.join(models_dir, fname), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"spec": dsp.bandit_spec(x)})[0]
        ours = dsp.bandit_ispec(out, 1, 2, L)
        rel = float(np.abs(ours - ref).max() / np.abs(ref).max())
        print(f"[reference] bandit {variant}: rel err vs original implementation {rel:.3e}")
        if rel > tol or not np.isfinite(ours).all():
            raise SystemExit(f"BandIt {variant} does not match the original implementation")
    print("OK: BandIt ONNX matches the original ZFTurbo/BandIt implementation")


if __name__ == "__main__":
    main()
