"""Audio decoding (any format FFmpeg understands, via PyAV) and WAV writing."""
import os
import wave

import numpy as np


def load_audio(path, samplerate=44100, start=None, end=None):
    """Decode `path` to float32 stereo [2, N] at `samplerate`.

    Only the [start, end] range (seconds, source time) is returned, which keeps
    work proportional to the clip actually used on the timeline.
    """
    import av

    with av.open(path) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError("No audio stream found in this file")
        stream = streams[0]
        stream.thread_type = "AUTO"
        seek_to = max(0.0, (start or 0.0) - 2.0)
        if seek_to > 0 and stream.time_base:
            container.seek(int(seek_to / stream.time_base), stream=stream, backward=True)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=samplerate)
        chunks = []
        t0 = None
        collected = 0
        need = None
        for frame in container.decode(stream):
            if t0 is None:
                t0 = float(frame.time) if frame.time is not None else 0.0
                if end is not None:
                    # stop decoding one second after the requested end
                    need = int(round((end - t0) * samplerate)) + samplerate
            for out in resampler.resample(frame):
                arr = out.to_ndarray()
                chunks.append(arr)
                collected += arr.shape[-1]
            if need is not None and collected >= need:
                break
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError("Could not decode any audio from this file")
    audio = np.concatenate(chunks, axis=-1).astype(np.float32)
    t0 = t0 or 0.0
    s = 0 if start is None else max(0, int(round((start - t0) * samplerate)))
    e = audio.shape[-1] if end is None else int(round((end - t0) * samplerate))
    audio = audio[:, s:e]
    if audio.shape[-1] == 0:
        raise ValueError("Selected range contains no audio")
    return np.ascontiguousarray(audio)


def write_wav(path, audio, samplerate=44100, bits=24):
    """Write [C, N] float audio as PCM WAV (16 or 24 bit)."""
    audio = np.clip(audio, -1.0, 1.0)
    inter = audio.T  # [N, C]
    tmp = path + ".part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(inter.shape[1])
        w.setframerate(samplerate)
        if bits == 16:
            w.setsampwidth(2)
            w.writeframes((inter * 32767.0).round().astype("<i2").tobytes())
        else:
            w.setsampwidth(3)
            ints = (inter * 8388607.0).round().astype("<i4")
            b = ints.reshape(-1, 1).view(np.uint8)[:, :3]
            w.writeframes(np.ascontiguousarray(b).tobytes())
    os.replace(tmp, path)


def varispeed(audio, samplerate, speed, target_len=None, reverse=False):
    """Play [C, N] audio at `speed` x (pitch follows speed, like Premiere's
    default speed change without "Maintain Audio Pitch").

    Implemented as a high-quality resample (libswresample): the samples are
    reinterpreted at samplerate*speed and converted back to samplerate.
    """
    import av

    if reverse:
        audio = audio[:, ::-1]
    if abs(speed - 1.0) > 1e-4:
        in_rate = max(1000, int(round(samplerate * speed)))
        layout = "stereo" if audio.shape[0] == 2 else "mono"
        res = av.AudioResampler(format="fltp", layout=layout, rate=samplerate)
        chunks = []
        block = 1 << 16
        for s in range(0, audio.shape[1], block):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(audio[:, s:s + block], dtype=np.float32),
                format="fltp", layout=layout)
            frame.sample_rate = in_rate
            for out in res.resample(frame):
                chunks.append(out.to_ndarray())
        for out in res.resample(None):
            chunks.append(out.to_ndarray())
        audio = np.concatenate(chunks, axis=-1)
    if target_len is not None:
        if audio.shape[1] >= target_len:
            audio = audio[:, :target_len]
        else:
            audio = np.pad(audio, ((0, 0), (0, target_len - audio.shape[1])))
    return np.ascontiguousarray(audio, dtype=np.float32)
