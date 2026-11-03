"""Build the standalone engine with PyInstaller and assemble the CEP extension.

    python engine/build_engine.py            -> dist/extension/<bundle id>/ (panel + bin/)
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLE = "com.bunnysframelab.audioseparator"
DIST = os.path.join(ROOT, "dist")


def main():
    work = os.path.join(ROOT, "build", "pyinstaller")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir", "--console", "--name", "bunnys-engine",
        "--paths", os.path.join(ROOT, "engine"),
        "--collect-all", "onnxruntime",
        "--collect-all", "av",
        "--exclude-module", "tkinter",
        "--exclude-module", "torch",
        "--exclude-module", "onnxruntime.transformers",
        "--exclude-module", "onnxruntime.quantization",
        "--exclude-module", "onnxruntime.tools",
        "--distpath", os.path.join(work, "dist"),
        "--workpath", os.path.join(work, "build"),
        "--specpath", work,
        os.path.join(ROOT, "engine", "main.py"),
    ])
    ext = os.path.join(DIST, "extension", BUNDLE)
    if os.path.exists(ext):
        shutil.rmtree(ext)
    shutil.copytree(os.path.join(ROOT, "panel"), ext)
    shutil.copytree(os.path.join(work, "dist", "bunnys-engine"),
                    os.path.join(ext, "bin", "bunnys-engine"))
    shutil.copy(os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), ext)
    print("extension ready:", ext)


if __name__ == "__main__":
    main()
