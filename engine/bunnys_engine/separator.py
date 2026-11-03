"""Source separation with ONNX Runtime.

* Picks the best execution provider for the machine (CUDA / DirectML / CoreML / CPU).
* Splits long audio into model-sized segments with overlap + triangular
  cross-fade (same scheme as demucs.apply.apply_model), batching segments on GPU.
* Falls back automatically: smaller GPU batch on out-of-memory, then CPU.
"""
import os

import numpy as np

from . import dsp

GPU_PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]

PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA CUDA",
    "DmlExecutionProvider": "GPU (DirectML)",
    "CoreMLExecutionProvider": "Apple GPU / Neural Engine (CoreML)",
    "CPUExecutionProvider": "CPU",
}

QUALITY = {
    # overlap between segments, number of random time shifts averaged
    "fast": {"overlap": 0.1, "shifts": 0},
    "balanced": {"overlap": 0.25, "shifts": 0},
    "best": {"overlap": 0.25, "shifts": 2},
}


def available_providers():
    import onnxruntime as ort
    return ort.get_available_providers()


def choose_providers(device="auto"):
    avail = available_providers()
    gpus = [p for p in GPU_PROVIDERS if p in avail]
    if device == "cpu" or not gpus:
        return ["CPUExecutionProvider"]
    return [gpus[0], "CPUExecutionProvider"]


def _padded(x, offset, length, target):
    """TensorChunk.padded from demucs: take real neighbouring audio as context."""
    total = x.shape[-1]
    delta = target - length
    start = offset - delta // 2
    end = start + target
    cs, ce = max(0, start), min(total, end)
    return np.pad(x[:, cs:ce], ((0, 0), (cs - start, end - ce)))


def _center_trim(x, length):
    delta = x.shape[-1] - length
    if delta:
        x = x[..., delta // 2: x.shape[-1] - (delta - delta // 2)]
    return x


class Separator:
    def __init__(self, model_path, info, device="auto", log=None, batch_size=None):
        self.model_path = model_path
        self.sources = list(info["sources"])
        self.samplerate = int(info["samplerate"])
        self.segment = int(info["segment_samples"])
        self.log = log or (lambda m: None)
        self.device = device
        self.providers = choose_providers(device)
        if device == "gpu" and self.providers[0] == "CPUExecutionProvider":
            self.log("No supported GPU found, using CPU instead")
        self._open(self.providers)
        on_gpu = self.providers[0] != "CPUExecutionProvider"
        self.batch_size = batch_size or (2 if on_gpu else 1)

    # -- session management -------------------------------------------------
    def _open(self, providers):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if providers[0] == "CPUExecutionProvider":
            so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        if providers[0] == "DmlExecutionProvider":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts = []
        for p in providers:
            if p == "CUDAExecutionProvider":
                opts.append({"cudnn_conv_algo_search": "HEURISTIC"})
            elif p == "CoreMLExecutionProvider":
                opts.append({"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"})
            else:
                opts.append({})
        try:
            self.session = ort.InferenceSession(self.model_path, so, providers=providers,
                                                provider_options=opts)
        except Exception as e:  # noqa: BLE001
            if providers[0] == "CPUExecutionProvider":
                raise
            self.log(f"{PROVIDER_LABELS.get(providers[0], providers[0])} could not start ({e}); using CPU")
            return self._open(["CPUExecutionProvider"])
        self.active_provider = self.session.get_providers()[0]
        self.providers = providers

    @property
    def device_label(self):
        return PROVIDER_LABELS.get(self.active_provider, self.active_provider)

    def _infer(self, batch):
        """batch: [B, 2, segment] -> [B, S, 2, segment]"""
        mag = dsp.model_spec(batch)
        spec_out, time_out = self.session.run(None, {"mix": batch, "mag": mag})
        return dsp.model_ispec(spec_out, batch.shape[-1]) + time_out

    def _infer_safe(self, batch):
        while True:
            try:
                return self._infer(batch)
            except Exception as e:  # noqa: BLE001
                if self.active_provider == "CPUExecutionProvider":
                    raise
                if len(batch) > 1:
                    # most likely out of GPU memory: process half at a time
                    self.batch_size = max(1, len(batch) // 2)
                    self.log(f"GPU busy/low memory, reducing batch to {self.batch_size}")
                    half = len(batch) // 2
                    return np.concatenate([self._infer_safe(batch[:half]),
                                           self._infer_safe(batch[half:])])
                self.log(f"GPU failed ({str(e)[:120]}); continuing on CPU")
                self._open(["CPUExecutionProvider"])

    # -- separation -------------------------------------------------------------
    def _split_apply(self, mix, overlap, progress):
        seg = self.segment
        length = mix.shape[-1]
        stride = max(1, int((1 - overlap) * seg))
        offsets = list(range(0, length, stride))
        weight = np.concatenate([np.arange(1, seg // 2 + 1),
                                 np.arange(seg - seg // 2, 0, -1)]).astype(np.float32)
        weight /= weight.max()
        out = np.zeros((len(self.sources), mix.shape[0], length), dtype=np.float32)
        sum_w = np.zeros(length, dtype=np.float32)
        done = 0
        i = 0
        while i < len(offsets):
            group = offsets[i:i + self.batch_size]
            lens = [min(length - o, seg) for o in group]
            batch = np.stack([_padded(mix, o, n, seg) for o, n in zip(group, lens)])
            res = self._infer_safe(np.ascontiguousarray(batch, dtype=np.float32))
            for o, n, r in zip(group, lens, res):
                w = weight[:n]
                out[..., o:o + n] += w * _center_trim(r, n)
                sum_w[o:o + n] += w
            i += len(group)
            done += len(group)
            progress(done / len(offsets))
        out /= sum_w
        return out

    def separate(self, wav, quality="balanced", progress=None):
        """wav: [2, N] float32 at self.samplerate -> dict source -> [2, N]"""
        progress = progress or (lambda f: None)
        q = QUALITY.get(quality, QUALITY["balanced"])
        ref = wav.mean(0)
        mean, std = float(ref.mean()), float(ref.std())
        if std < 1e-8:
            return {s: np.zeros_like(wav) for s in self.sources}
        mix = (wav - mean) / std
        shifts = q["shifts"]
        if not shifts:
            est = self._split_apply(mix, q["overlap"], progress)
        else:
            max_shift = int(0.5 * self.samplerate)
            length = mix.shape[-1]
            padded = np.pad(mix, ((0, 0), (max_shift, max_shift)))
            rng = np.random.default_rng(0)
            est = 0
            for k in range(shifts):
                off = int(rng.integers(0, max_shift))
                chunk = padded[:, off:off + length + max_shift - off]
                res = self._split_apply(
                    chunk, q["overlap"], lambda f, k=k: progress((k + f) / shifts))
                est = est + res[..., max_shift - off:max_shift - off + length]
            est = est / shifts
        est = est * std + mean
        return {s: est[i] for i, s in enumerate(self.sources)}


# ----------------------------------------------------------------------------
# Output stems per mode
# ----------------------------------------------------------------------------
MUSIC_STEMS = {
    "vocals": ("Vocals", ["vocals"]),
    "instrumental": ("Instrumental", ["drums", "bass", "guitar", "piano", "other"]),
    "drums": ("Drums", ["drums"]),
    "bass": ("Bass", ["bass"]),
    "guitar": ("Guitar", ["guitar"]),
    "piano": ("Piano", ["piano"]),
    "other": ("Other", ["other"]),
}

# Experimental: derived from the 6-stem music model until a dedicated
# dialogue/effects/music model is integrated.
CINEMATIC_STEMS = {
    "dialogue": ("Vocal-Dialogue", ["vocals"]),
    "sfx": ("SFX", ["other"]),
    "music": ("Music", ["drums", "bass", "guitar", "piano"]),
}

MODES = {"music": MUSIC_STEMS, "cinematic": CINEMATIC_STEMS}


def build_stems(sources, mode, wanted):
    table = MODES[mode]
    result = []
    for key in wanted:
        if key not in table:
            raise ValueError(f"Unknown stem '{key}' for mode '{mode}'")
        label, parts = table[key]
        result.append((key, label, sum(sources[p] for p in parts)))
    return result

