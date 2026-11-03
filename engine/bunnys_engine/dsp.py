"""NumPy spectrogram transforms that match HTDemucs (_spec / _ispec) bit-for-bit
up to float rounding. Kept outside the ONNX graph on purpose: STFT/iSTFT on
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


def _reflect_pad(x, left, right):
    pad = [(0, 0)] * (x.ndim - 1) + [(left, right)]
    return np.pad(x, pad, mode="reflect")


def stft(x, n_fft=NFFT, hop=HOP):
    """torch.stft(center=True, pad_mode='reflect', normalized=True, hann)."""
    win = _hann(n_fft)
    xp = _reflect_pad(x, n_fft // 2, n_fft // 2)
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft, axis=-1)[..., ::hop, :]
    spec = np.fft.rfft(frames * win, axis=-1).astype(np.complex64)
    spec *= np.float32(1.0 / math.sqrt(n_fft))
    return np.swapaxes(spec, -1, -2)  # [..., freqs, frames]


def istft(z, length, n_fft=NFFT, hop=HOP):
    """torch.istft(center=True, normalized=True, hann, length=length)."""
    win = _hann(n_fft)
    n_frames = z.shape[-1]
    zz = np.swapaxes(z, -1, -2) * np.float32(math.sqrt(n_fft))
    frames = np.fft.irfft(zz, n=n_fft, axis=-1).astype(np.float32) * win
    total = n_fft + hop * (n_frames - 1)
    lead = frames.shape[:-2]
    out = np.zeros(lead + (total,), dtype=np.float32)
    env = np.zeros(total, dtype=np.float32)
    win_sq = win * win
    for i in range(n_frames):
        s = i * hop
        out[..., s:s + n_fft] += frames[..., i, :]
        env[s:s + n_fft] += win_sq
    start = n_fft // 2
    out = out[..., start:start + length]
    env = env[start:start + length]
    out /= np.where(env > 1e-11, env, 1.0)
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
