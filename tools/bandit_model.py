"""ONNX-friendly re-implementation of BandIt Plus (multi-mask band-split RNN).

Architecture and parameter names follow the MIT-licensed implementation in
ZFTurbo/Music-Source-Separation-Training (models/bandit, originally by
Karn Watcharasupat et al., "A Generalized Bandsplit Neural Network for
Cinematic Audio Source Separation"). Parameter names match that code, so its
checkpoints load directly.

Differences, for portability only (the maths is identical):
* complex tensors are carried as a trailing real/imag axis of size 2;
* the STFT / iSTFT live outside the network (engine/bunnys_engine/dsp.py);
* in-place scatter ops are replaced by stack / pad.

    input : spec  [N, 1, F, T, 2]   (N = batch * channels; mono per item)
    output: stems [N, S, F, T, 2]
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# band layout ("musical" filterbank)
# ---------------------------------------------------------------------------
def _hz_to_midi(f):
    return 12.0 * (np.log2(np.asarray(f, dtype=np.float64)) - np.log2(440.0)) + 69.0


def _midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def musical_band_specs(n_bands, fs, n_fft):
    n_freqs = n_fft // 2 + 1
    df = fs / n_fft
    f_max = fs / 2
    f_min = fs / n_fft
    n_octaves = np.log2(f_max / f_min)
    bandwidth_mult = np.power(2.0, n_octaves / n_bands)
    low_midi = max(0, _hz_to_midi(f_min))
    high_midi = _hz_to_midi(f_max)
    hz_pts = _midi_to_hz(np.linspace(low_midi, high_midi, n_bands))
    low_bins = np.floor(hz_pts / bandwidth_mult / df).astype(int)
    high_bins = np.ceil(hz_pts * bandwidth_mult / df).astype(int)
    fb = np.zeros((n_bands, n_freqs))
    for i in range(n_bands):
        fb[i, low_bins[i]:high_bins[i] + 1] = 1.0
    fb[0, :low_bins[0]] = 1.0
    fb[-1, high_bins[-1] + 1:] = 1.0
    fb = torch.as_tensor(fb)
    norm = fb / fb.sum(dim=0, keepdim=True)
    specs, weights = [], []
    for i in range(n_bands):
        active = torch.nonzero(fb[i, :]).squeeze().tolist()
        if isinstance(active, int):
            active = (active, active)
        if len(active) == 0:
            continue
        s, e = active[0], active[-1] + 1
        specs.append((s, e))
        weights.append(norm[i, s:e])
    return specs, weights


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------
class NormFC(nn.Module):
    def __init__(self, emb_dim, bandwidth, in_channel):
        super().__init__()
        self.norm = nn.LayerNorm(in_channel * bandwidth * 2)
        self.fc = nn.Linear(in_channel * bandwidth * 2, emb_dim)

    def forward(self, xb):  # (b, t, in_chan, 2*bw)
        b, t, c, ribw = xb.shape
        return self.fc(self.norm(xb.reshape(b, t, c * ribw)))


class BandSplit(nn.Module):
    def __init__(self, band_specs, emb_dim, in_channel):
        super().__init__()
        self.band_specs = band_specs
        self.norm_fc_modules = nn.ModuleList(
            [NormFC(emb_dim, e - s, in_channel) for s, e in band_specs])

    def forward(self, xr):  # (b, in_chan, F, T, 2)
        x = xr.permute(0, 3, 1, 4, 2)  # (b, t, in_chan, 2, F)
        b, t, c = x.shape[0], x.shape[1], x.shape[2]
        z = [nfm(x[..., s:e].reshape(b, t, c, -1)) for (s, e), nfm in
             zip(self.band_specs, self.norm_fc_modules)]
        return torch.stack(z, dim=1)  # (b, n_bands, t, emb)


class ResidualRNN(nn.Module):
    def __init__(self, emb_dim, rnn_dim, rnn_type="GRU"):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = cls(input_size=emb_dim, hidden_size=rnn_dim, num_layers=1,
                       batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_dim * 2, emb_dim)

    def forward(self, z):
        z0 = z
        z = self.norm(z)
        b, n, a, e = z.shape
        z = self.rnn(z.reshape(b * n, a, e))[0]
        z = self.fc(z.reshape(b, n, a, -1))
        return z + z0


class SeqBand(nn.Module):
    def __init__(self, n_modules, emb_dim, rnn_dim, rnn_type):
        super().__init__()
        self.seqband = nn.ModuleList(
            [ResidualRNN(emb_dim, rnn_dim, rnn_type) for _ in range(2 * n_modules)])

    def forward(self, z):
        for m in self.seqband:
            z = m(z).transpose(1, 2)
        return z


class NormMLP(nn.Module):
    def __init__(self, emb_dim, mlp_dim, bandwidth, in_channel):
        super().__init__()
        self.bandwidth = bandwidth
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = nn.Sequential(nn.Linear(emb_dim, mlp_dim), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(mlp_dim, bandwidth * in_channel * 2 * 2), nn.GLU(dim=-1))

    def forward(self, qb):  # (b, t, emb)
        mb = self.output(self.hidden(self.norm(qb)))
        b, t, _ = mb.shape
        mb = mb.reshape(b, t, self.in_channel, self.bandwidth, 2)
        return mb.permute(0, 2, 3, 1, 4)  # (b, in_chan, bw, t, 2)


class OverlappingMaskEstimation(nn.Module):
    def __init__(self, band_specs, freq_weights, n_freq, emb_dim, mlp_dim, in_channel):
        super().__init__()
        self.band_specs = band_specs
        self.n_freq = n_freq
        self.norm_mlp = nn.ModuleList(
            [NormMLP(emb_dim, mlp_dim, e - s, in_channel) for s, e in band_specs])
        for i, fw in enumerate(freq_weights):
            self.register_buffer(f"freq_weights/{i}", fw.float())

    def forward(self, q):  # (b, n_bands, t, emb)
        total = None
        for i, ((s, e), nmlp) in enumerate(zip(self.band_specs, self.norm_mlp)):
            m = nmlp(q[:, i])
            m = m * self.get_buffer(f"freq_weights/{i}")[:, None, None]
            m = F.pad(m, (0, 0, 0, 0, s, self.n_freq - e))
            total = m if total is None else total + m
        return total  # (b, in_chan, F, t, 2)


class _Core(nn.Module):
    def __init__(self, stems, band_specs, freq_weights, n_freq, in_channel,
                 n_sqm_modules, emb_dim, rnn_dim, rnn_type, mlp_dim):
        super().__init__()
        self.band_split = BandSplit(band_specs, emb_dim, in_channel)
        self.tf_model = SeqBand(n_sqm_modules, emb_dim, rnn_dim, rnn_type)
        self.mask_estim = nn.ModuleDict({
            s: OverlappingMaskEstimation(band_specs, freq_weights, n_freq, emb_dim, mlp_dim, in_channel)
            for s in stems})


class BanditPlus(nn.Module):
    def __init__(self, stems=("speech", "music", "effects"), n_bands=64, fs=44100, n_fft=2048,
                 in_channel=1, n_sqm_modules=8, emb_dim=128, rnn_dim=256, rnn_type="GRU",
                 mlp_dim=512, **_ignored):
        super().__init__()
        self.stems = list(stems)
        specs, weights = musical_band_specs(n_bands, fs, n_fft)
        self.bsrnn = _Core(self.stems, specs, weights, n_fft // 2 + 1, in_channel,
                           n_sqm_modules, emb_dim, rnn_dim, rnn_type, mlp_dim)

    def forward(self, spec):  # (N, 1, F, T, 2)
        core = self.bsrnn
        q = core.tf_model(core.band_split(spec))
        xr, xi = spec[..., 0], spec[..., 1]
        outs = []
        for stem in self.stems:
            m = core.mask_estim[stem](q)
            mr, mi = m[..., 0], m[..., 1]
            outs.append(torch.stack([xr * mr - xi * mi, xr * mi + xi * mr], dim=-1)[:, 0])
        return torch.stack(outs, dim=1)  # (N, S, F, T, 2)


def load_bandit(config_model, checkpoint_path):
    """Build from the BandIt config's `model:` section and load weights safely."""
    model = BanditPlus(**config_model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k not in keep]
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
    for k, v in keep.items():
        if k.split("/")[0].endswith("freq_weights") and not torch.allclose(own[k].float(), v.float(), atol=1e-6):
            raise RuntimeError(f"band layout mismatch at {k}")
    model.load_state_dict(keep, strict=True)
    return model.eval()
