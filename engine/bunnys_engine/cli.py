"""Command line entry used by the Premiere Pro panel.

Every message is one JSON object per line on stdout, so the panel can show
live progress:  {"type": "progress", "value": 0.42, "stage": "separate"}
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

from . import __version__

STAGES = {"download": (0.00, 0.10), "decode": (0.10, 0.15),
          "separate": (0.15, 0.95), "write": (0.95, 1.00)}


def emit(kind, **data):
    data["type"] = kind
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def progress(stage, frac):
    a, b = STAGES[stage]
    emit("progress", value=round(a + (b - a) * min(max(frac, 0.0), 1.0), 4), stage=stage)


def log(msg):
    emit("status", message=msg)


def safe_name(name):
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")
    return name[:80] or "audio"


def cmd_info(args):
    from .separator import available_providers, choose_providers, PROVIDER_LABELS
    from .models import default_models_dir
    providers = available_providers()
    best = choose_providers("auto")[0]
    emit("info", version=__version__, providers=providers,
         best=PROVIDER_LABELS.get(best, best), gpu=best != "CPUExecutionProvider",
         models_dir=default_models_dir())


def cmd_separate(args):
    from .audio_io import load_audio, write_wav, varispeed
    from .models import ensure_model, model_info
    from .separator import Separator, build_stems, MODES

    t_start = time.time()
    model_name, table = MODES[args.mode]
    wanted = [s.strip().lower() for s in args.stems.split(",") if s.strip()]
    wanted = [w for w in dict.fromkeys(wanted)] or list(table.keys())[:3]

    log("Preparing AI model...")
    info = model_info(model_name, args.models_dir)

    def resolve(variant):
        path = ensure_model(model_name, variant, args.models_dir,
                            progress=lambda f: progress("download", f), log=log)
        progress("download", 1)
        return path

    sep = Separator(info, resolve, device=args.device, log=log)
    emit("device", label=sep.device_label, provider=sep.active_provider)
    log(f"Running on {sep.device_label}")

    log("Reading audio...")
    wav = load_audio(args.input, sep.samplerate, args.start, args.end)
    progress("decode", 1)
    duration = wav.shape[-1] / sep.samplerate

    log(f"Separating {duration:.1f}s of audio...")
    sources = sep.separate(wav, args.quality, progress=lambda f: progress("separate", f))

    os.makedirs(args.out_dir, exist_ok=True)
    base = safe_name(args.name or os.path.splitext(os.path.basename(args.input))[0])
    stems = build_stems(sources, args.mode, wanted)
    target = None
    if args.timeline_duration:
        target = int(round(args.timeline_duration * sep.samplerate))
    out_seconds = duration
    files = []
    for i, (key, label, audio) in enumerate(stems):
        if args.speed != 1.0 or args.reverse or target is not None:
            audio = varispeed(audio, sep.samplerate, args.speed, target, args.reverse)
            out_seconds = audio.shape[-1] / sep.samplerate
        path = os.path.join(args.out_dir, f"{base} - {label}.wav")
        write_wav(path, audio, sep.samplerate, bits=args.bits)
        files.append({"stem": key, "label": label, "path": path})
        progress("write", (i + 1) / len(stems))

    elapsed = time.time() - t_start
    emit("done", files=files, seconds=round(elapsed, 2), audio_seconds=round(out_seconds, 4),
         source_seconds=round(duration, 4), device=sep.device_label)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="bunnys-engine", description="Bunnys Audio Separator engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sp = sub.add_parser("separate")
    sp.add_argument("--input", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--name")
    sp.add_argument("--start", type=float)
    sp.add_argument("--end", type=float)
    sp.add_argument("--mode", choices=["music", "cinematic"], default="music")
    sp.add_argument("--stems", default="")
    sp.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    sp.add_argument("--quality", choices=["fast", "balanced", "best"], default="balanced")
    sp.add_argument("--bits", type=int, choices=[16, 24], default=24)
    sp.add_argument("--speed", type=float, default=1.0,
                    help="clip speed on the timeline (2.0 = 200%%); stems are re-timed to match")
    sp.add_argument("--reverse", action="store_true", help="clip plays reversed on the timeline")
    sp.add_argument("--timeline-duration", type=float,
                    help="exact duration of the clip on the timeline, in seconds")
    sp.add_argument("--models-dir")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "info":
            cmd_info(args)
        else:
            cmd_separate(args)
    except Exception as e:  # noqa: BLE001
        emit("error", message=str(e), detail=traceback.format_exc()[-2000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
