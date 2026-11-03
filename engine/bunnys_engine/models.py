"""Model files: located in a per-user cache and downloaded on first use.

Models are published as assets of the `models-v1` GitHub release, so the
installer stays small and users only download what they actually use.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = "uday629/bunnys-audio-separator"
MODELS_TAG = "models-v1"
BASE_URL = os.environ.get(
    "BUNNYS_MODELS_URL",
    f"https://github.com/{REPO}/releases/download/{MODELS_TAG}",
)


def default_models_dir():
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(root, "BunnysFramelab", "AudioSeparator", "models")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/BunnysFramelab/AudioSeparator/models")
    return os.path.expanduser("~/.local/share/bunnys-audio-separator/models")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url, dest, progress=None):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "BunnysAudioSeparator"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if progress and total:
                progress(done / total)
    os.replace(tmp, dest)


def load_manifest(models_dir, progress=None):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "models.json")
    if not os.path.exists(path):
        _download(f"{BASE_URL}/models.json", path)
    with open(path) as f:
        return json.load(f)


def ensure_model(name, variant="fp32", models_dir=None, progress=None, log=None):
    """Return (path, info) for a model file, downloading + verifying if needed."""
    models_dir = models_dir or default_models_dir()
    manifest = load_manifest(models_dir)
    info = manifest["models"][name]
    var = info["variants"].get(variant) or info["variants"]["fp32"]
    path = os.path.join(models_dir, var["file"])
    if os.path.exists(path) and os.path.getsize(path) == var["size"]:
        return path, info
    if log:
        log(f"Downloading AI model ({var['size'] / 1e6:.0f} MB, one time only)...")
    _download(f"{BASE_URL}/{var['file']}", path, progress)
    if _sha256(path) != var["sha256"]:
        os.remove(path)
        raise RuntimeError("Model download was corrupted, please try again")
    return path, info
