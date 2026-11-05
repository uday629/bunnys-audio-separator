"""Bundle an installer with its uninstaller, README and notices into a ZIP.

    python tools/package_release.py --platform windows --installer dist/X.exe --version 0.2.0 --label Windows
"""
import argparse
import base64
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

UNINSTALLERS = {
    "windows": "installer/windows/Uninstall Bunnys Audio Separator.bat",
    "macos": "installer/macos/Uninstall Bunnys Audio Separator.command",
}


def add(z, arcname, data, executable=False):
    info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if executable else 0o644) | 0o100000) << 16
    z.writestr(info, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["windows", "macos"], required=True)
    ap.add_argument("--installer", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--label", required=True, help="e.g. Windows, macOS-AppleSilicon")
    args = ap.parse_args()

    folder = f"BunnysAudioSeparator-v{args.version}-{args.label}"
    out = os.path.join(ROOT, "dist", folder + ".zip")

    with open(os.path.join(ROOT, "panel/assets/logo.png"), "rb") as f:
        logo = base64.b64encode(f.read()).decode()
    with open(os.path.join(ROOT, "installer/README.template.html"), encoding="utf-8") as f:
        readme = f.read().replace("{{LOGO}}", logo).replace("{{VERSION}}", args.version)

    ext = os.path.splitext(args.installer)[1]
    with zipfile.ZipFile(out, "w") as z:
        with open(args.installer, "rb") as f:
            add(z, f"{folder}/BunnysAudioSeparator-Setup-v{args.version}-{args.label}{ext}", f.read())
        un = UNINSTALLERS[args.platform]
        with open(os.path.join(ROOT, un), "rb") as f:
            data = f.read()
        if args.platform == "windows":
            data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        add(z, f"{folder}/{os.path.basename(un)}", data, executable=args.platform == "macos")
        add(z, f"{folder}/README.html", readme.encode("utf-8"))
        with open(os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), "rb") as f:
            add(z, f"{folder}/THIRD_PARTY_NOTICES.md", f.read())
    print("wrote", out, os.path.getsize(out) // 1_000_000, "MB")


if __name__ == "__main__":
    main()
