"""NumPy spectrogram transforms matching HTDemucs (_spec/_ispec) and BandIt
(torchaudio Spectrogram/InverseSpectrogram) up to float rounding. Kept outside the ONNX graph on purpose: STFT/iSTFT on
complex tensors is not portable across ONNX Runtime execution providers.
"""
import math

import numpy as np

NFFT = 4096
HOP = 1024

_WINDOW_CACHE = {}


def _hann(n):
    w = _WINDOW_CACHE.get(n)
    if w is None:
        # torch.hann_window default is periodic=True
        w = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)
        _WINDOW_CACHE[n] = w
    return w


def _pad(x, left, right, mode="reflect"):
    pad = [(0, 0)] * (x.ndim - 1) + [(left, right)]
    return np.pad(x, pad, mode=mode)


# kept for model_spec below
_reflect_pad = _pad


def stft(x, n_fft=NFFT, hop=HOP, pad_mode="reflect", scale=None):
    """torch.stft(center=True, hann, return_complex) -> [..., freqs, frames].

    scale defaults to 1/sqrt(n_fft) (torch `normalized=True`); torchaudio's
    `normalized=True` corresponds to scale = 1/sqrt(sum(window**2)).
    """
    win = _hann(n_fft)
    xp = _pad(x, n_fft // 2, n_fft // 2, "constant" if pad_mode == "constant" else "reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft, axis=-1)[..., ::hop, :]
    spec = np.fft.rfft(frames * win, axis=-1).astype(np.complex64)
    spec *= np.float32(scale if scale is not None else 1.0 / math.sqrt(n_fft))
    return np.swapaxes(spec, -1, -2)


_ENV_CACHE = {}


def _ola_envelope(n_fft, hop, n_frames):
    key = (n_fft, hop, n_frames)
    env = _ENV_CACHE.get(key)
    if env is None:
        win_sq = _hann(n_fft) ** 2
        env = np.zeros(n_fft + hop * (n_frames - 1), dtype=np.float32)
        for i in range(n_frames):
            env[i * hop:i * hop + n_fft] += win_sq
        _ENV_CACHE[key] = env
    return env


def istft(z, length, n_fft=NFFT, hop=HOP, scale=None):
    """torch.istft(center=True, hann, length=length); scale is the inverse of stft's."""
    win = _hann(n_fft)
    n_frames = z.shape[-1]
    zz = np.swapaxes(z, -1, -2) * np.float32(scale if scale is not None else math.sqrt(n_fft))
    frames = np.fft.irfft(zz, n=n_fft, axis=-1).astype(np.float32) * win
    lead = frames.shape[:-2]
    total = n_fft + hop * (n_frames - 1)
    if n_fft % hop == 0:
        # vectorised overlap-add: each frame spans k consecutive hop-sized blocks
        k = n_fft // hop
        blocks = np.zeros(lead + (n_frames + k - 1, hop), dtype=np.float32)
        fr = frames.reshape(lead + (n_frames, k, hop))
        for j in range(k):
            blocks[..., j:j + n_frames, :] += fr[..., :, j, :]
        out = blocks.reshape(lead + (total,))
    else:
        out = np.zeros(lead + (total,), dtype=np.float32)
        for i in range(n_frames):
            out[..., i * hop:i * hop + n_fft] += frames[..., i, :]
    env = _ola_envelope(n_fft, hop, n_frames)
    start = n_fft // 2
    out = out[..., start:start + length]
    env = env[start:start + length]
    out = out / np.where(env > 1e-11, env, 1.0)
    if out.shape[-1] < length:
        pad = [(0, 0)] * (out.ndim - 1) + [(0, length - out.shape[-1])]
        out = np.pad(out, pad)
    return out


def model_spec(mix):
    """HTDemucs._spec + _magnitude(cac). mix: [B, C, L] -> mag [B, 2C, F, T]."""
    hl = HOP
    L = mix.shape[-1]
    le = int(math.ceil(L / hl))
    pad = hl // 2 * 3
    x = _reflect_pad(mix, pad, pad + le * hl - L)
    z = stft(x)[..., :-1, :]
    z = z[..., 2:2 + le]
    B, C, Fr, T = z.shape
    mag = np.stack([z.real, z.imag], axis=2)  # [B, C, 2, F, T]
    return np.ascontiguousarray(mag.reshape(B, C * 2, Fr, T), dtype=np.float32)


def model_ispec(spec_out, length):
    """HTDemucs._mask(cac) + _ispec. spec_out: [B, S, 2C, F, T] -> [B, S, C, length]."""
    B, S, C2, Fr, T = spec_out.shape
    m = spec_out.reshape(B, S, C2 // 2, 2, Fr, T)
    z = m[:, :, :, 0] + 1j * m[:, :, :, 1]
    z = z.astype(np.complex64)
    hl = HOP
    z = np.pad(z, [(0, 0)] * 3 + [(0, 1), (2, 2)])
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = istft(z, le)
    return x[..., pad:pad + length]


# ---------------------------------------------------------------------------
# BandIt (torchaudio, n_fft 2048, hop 512, zero padding, window-normalised)
# ---------------------------------------------------------------------------
BANDIT_NFFT = 2048
BANDIT_HOP = 512


def _bandit_scale():
    return 1.0 / math.sqrt(float((_hann(BANDIT_NFFT).astype(np.float64) ** 2).sum()))


def bandit_spec(mix):
    """mix [B, C, L] -> [B*C, 1, F, T, 2] real/imag spectrogram (BandIt input)."""
    B, C, L = mix.shape
    z = stft(mix.reshape(B * C, L), BANDIT_NFFT, BANDIT_HOP, "constant", _bandit_scale())
    out = np.stack([z.real, z.imag], axis=-1)[:, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def bandit_ispec(spec_out, batch, channels, length):
    """spec_out [B*C, S, F, T, 2] -> [B, S, C, length]."""
    N, S, Fr, T, _ = spec_out.shape
    z = (spec_out[..., 0] + 1j * spec_out[..., 1]).astype(np.complex64)
    x = istft(z, length, BANDIT_NFFT, BANDIT_HOP, 1.0 / _bandit_scale())  # [N, S, length]
    return np.ascontiguousarray(x.reshape(batch, channels, S, length).transpose(0, 2, 1, 3))
