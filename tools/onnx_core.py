"""ONNX-exportable core of HTDemucs.

torch.onnx cannot export torch.stft / torch.istft with complex outputs, so the
spectrogram transforms are moved out of the graph and done in NumPy by the
engine (see engine/bunnys_engine/dsp.py). This module wraps a pretrained
HTDemucs and exposes only the neural network part:

    inputs : mix [B, 2, L]  (time domain, L = training length)
             mag [B, 4, F, T] (complex-as-channels spectrogram of `mix`)
    outputs: spec_out [B, S, 4, F, T] (complex-as-channels spectrogram per source)
             time_out [B, S, 2, L]    (time-branch waveform per source)

The final waveform per source is  istft(spec_out) + time_out , exactly as in
HTDemucs.forward.
"""
import torch
from torch import nn
from einops import rearrange


class HTDemucsCore(nn.Module):
    def __init__(self, model, half=False):
        super().__init__()
        self.m = model
        self.half_precision = half
        assert model.cac, "only complex-as-channels HTDemucs is supported"

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        if self.half_precision:
            # statistics above stay fp32; the network itself runs in fp16
            x = x.half()
            xt = xt.half()

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        if self.half_precision:
            x = x.float()
            xt = xt.float()

        S = len(m.sources)
        L = mix.shape[-1]
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, L)
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


class half_positional_embeddings:
    """Context manager: make demucs' sinusoidal embeddings follow fp16 weights.

    demucs.transformer builds its position embeddings in fp32, which silently
    promotes the whole cross-transformer back to fp32 when the model is .half().
    """

    def __enter__(self):
        import demucs.transformer as T
        self.T = T
        self.saved = (T.create_sin_embedding, T.create_2d_sin_embedding)
        a, b = self.saved
        T.create_sin_embedding = lambda *x, **k: a(*x, **k).half()
        T.create_2d_sin_embedding = lambda *x, **k: b(*x, **k).half()
        return self

    def __exit__(self, *exc):
        self.T.create_sin_embedding, self.T.create_2d_sin_embedding = self.saved
