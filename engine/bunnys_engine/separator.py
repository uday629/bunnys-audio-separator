"""Source separation with ONNX Runtime.

* Picks the best execution provider for the machine (CUDA / DirectML / CoreML / CPU)
  and the matching model precision (fp16 on CUDA/DirectML, fp32 elsewhere).
* Splits long audio into model-sized segments with overlap + triangular
  cross-fade (same scheme as demucs.apply.apply_model), batching segments on GPU.
* Never gives up silently: low GPU memory -> smaller batches; numerical trouble
  in fp16 -> fp32; GPU failure -> CPU.
"""
import os

import numpy as np

from . import dsp

GPU_PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]
FP16_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")

PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA CUDA",
    "DmlExecutionProvider": "GPU (DirectML)",
    "CoreMLExecutionProvider": "Apple GPU / Neural Engine (CoreML)",
    "CPUExecutionProvider": "CPU",
}

# overlap between segments, number of random time shifts averaged
QUALITY = {
    "htdemucs": {
        "fast": {"overlap": 0.1, "shifts": 0},
        "balanced": {"overlap": 0.25, "shifts": 0},
        "best": {"overlap": 0.25, "shifts": 2},
    },
    "bandit": {
        "fast": {"overlap": 0.1, "shifts": 0},
        "balanced": {"overlap": 0.25, "shifts": 0},
        "best": {"overlap": 0.5, "shifts": 0},
    },
}


class NonFiniteOutput(RuntimeError):
    pass


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
    """resolve_model(variant) must return a local path for "fp32"/"fp16"."""

    def __init__(self, info, resolve_model, device="auto", log=None, batch_size=None):
        self.kind = info.get("kind", "htdemucs")
        self.sources = list(info["sources"])
        self.samplerate = int(info["samplerate"])
        self.segment = int(info["segment_samples"])
        self.normalize = bool(info.get("normalize", self.kind == "htdemucs"))
        self.variants = set(info["variants"])
        self._resolve = resolve_model
        self.log = log or (lambda m: None)
        providers = choose_providers(device)
        if device == "gpu" and providers[0] == "CPUExecutionProvider":
            self.log("No supported GPU found, using CPU instead")
        self._open(providers)
        on_gpu = self.active_provider != "CPUExecutionProvider"
        self.batch_size = batch_size or (2 if on_gpu else 1)

    # -- session management -------------------------------------------------
    def _pick_variant(self, provider):
        return "fp16" if provider in FP16_PROVIDERS and "fp16" in self.variants else "fp32"

    def _open(self, providers, variant=None):
        import onnxruntime as ort
        variant = variant or self._pick_variant(providers[0])
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
            path = self._resolve(variant)
            self.session = ort.InferenceSession(path, so, providers=providers, provider_options=opts)
        except Exception as e:  # noqa: BLE001
            if providers[0] == "CPUExecutionProvider" and variant == "fp32":
                raise
            if variant == "fp16":
                self.log(f"Fast fp16 model unavailable ({str(e)[:100]}); using fp32")
                return self._open(providers, "fp32")
            self.log(f"{PROVIDER_LABELS.get(providers[0], providers[0])} could not start "
                     f"({str(e)[:120]}); using CPU")
            return self._open(["CPUExecutionProvider"], "fp32")
        self.providers = providers
        self.variant = variant
        self.active_provider = self.session.get_providers()[0]

    @property
    def device_label(self):
        label = PROVIDER_LABELS.get(self.active_provider, self.active_provider)
        return label + (" · fp16" if self.variant == "fp16" else "")

    def _run(self, batch):
        """batch: [B, 2, segment] -> [B, S, 2, segment]"""
        if self.kind == "bandit":
            B, C, L = batch.shape
            out = self.session.run(None, {"spec": dsp.bandit_spec(batch)})[0]
            res = dsp.bandit_ispec(out, B, C, L)
        else:
            mag = dsp.model_spec(batch)
            spec_out, time_out = self.session.run(None, {"mix": batch, "mag": mag})
            res = dsp.model_ispec(spec_out, batch.shape[-1]) + time_out
        return res

    def _infer(self, batch):
        res = self._run(batch)
        if not np.isfinite(res).all():
            raise NonFiniteOutput("model produced invalid numbers")
        return res

    def _infer_safe(self, batch):
        while True:
            try:
                return self._infer(batch)
            except NonFiniteOutput:
                if self.variant == "fp16":
                    self.log("fp16 precision issue on this GPU; switching to fp32")
                    self._open(self.providers, "fp32")
                    continue
                if self.active_provider != "CPUExecutionProvider":
                    self.log("GPU returned invalid output; continuing on CPU")
                    self._open(["CPUExecutionProvider"], "fp32")
                    continue
                return np.nan_to_num(self._run(batch))
            except Exception as e:  # noqa: BLE001
                if self.active_provider == "CPUExecutionProvider":
                    raise
                if len(batch) > 1:
                    # most likely out of GPU memory: process half at a time from now on
                    self.batch_size = max(1, len(batch) // 2)
                    self.log(f"GPU memory is tight, reducing batch to {self.batch_size}")
                    half = len(batch) // 2
                    return np.concatenate([self._infer_safe(batch[:half]),
                                           self._infer_safe(batch[half:])])
                self.log(f"GPU failed ({str(e)[:120]}); continuing on CPU")
                self._open(["CPUExecutionProvider"], "fp32")

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
        out /= np.maximum(sum_w, 1e-8)
        return out

    def separate(self, wav, quality="balanced", progress=None):
        """wav: [2, N] float32 at self.samplerate -> dict source -> [2, N]"""
        progress = progress or (lambda f: None)
        q = QUALITY[self.kind].get(quality, QUALITY[self.kind]["balanced"])
        ref = wav.mean(0)
        mean, std = float(ref.mean()), float(ref.std())
        if std < 1e-8:
            progress(1.0)
            return {s: np.zeros_like(wav) for s in self.sources}
        if self.normalize:
            mix = (wav - mean) / std
        else:
            mix = wav
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
        if self.normalize:
            est = est * std + mean
        return {s: est[i] for i, s in enumerate(self.sources)}


# ----------------------------------------------------------------------------
# Output stems per mode:  key -> (label, model, [model sources summed])
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

CINEMATIC_STEMS = {
    "dialogue": ("Vocal-Dialogue", ["speech"]),
    "sfx": ("SFX", ["effects"]),
    "music": ("Music", ["music"]),
    "background": ("Background (Music+SFX)", ["music", "effects"]),
}

MODES = {
    "music": ("htdemucs_6s", MUSIC_STEMS),
    "cinematic": ("bandit_dnr", CINEMATIC_STEMS),
}


def build_stems(sources, mode, wanted):
    table = MODES[mode][1]
    result = []
    for key in wanted:
        if key not in table:
            raise ValueError(f"Unknown stem '{key}' for mode '{mode}'")
        label, parts = table[key]
        result.append((key, label, sum(sources[p] for p in parts)))
    return result
