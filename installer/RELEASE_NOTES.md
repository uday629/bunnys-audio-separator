## Bunnys Audio Separator — by Bunnys Framelab

AI stem separation inside Adobe Premiere Pro (2019 and newer), on Windows and macOS.

### Download — one ZIP for your computer
| Your computer | Download |
|---|---|
| Windows 10/11 (64-bit) | `BunnysAudioSeparator-v…-Windows.zip` |
| Mac with Apple Silicon (M1/M2/M3/M4) | `BunnysAudioSeparator-v…-macOS-AppleSilicon.zip` |
| Mac with Intel processor | `BunnysAudioSeparator-v…-macOS-Intel.zip` |

Each ZIP contains the **installer**, an **uninstaller**, a **README.html** (install · use · uninstall, English + বাংলা) and the open-source license notices.

Unzip it, run the installer, restart Premiere Pro and open **Window → Extensions → Bunnys Audio Separator**.
The AI models download automatically the first time each mode is used.

**The installers are not code-signed yet.**
- Windows: if you see "Windows protected your PC", click **More info → Run anyway**.
- macOS: if the package is blocked, open **System Settings → Privacy & Security** and click **Open Anyway**.

### What's new in 0.2
- **Vocal · SFX · Music now uses a dedicated AI model** (BandIt Plus, trained for dialogue / effects / music) instead of the music model — much cleaner dialogue and effects. New **Background** stem (music + SFX without the voice) and a "Remove voice" preset.
- **Faster on GPUs:** half-precision (fp16) models on NVIDIA/AMD/Intel GPUs (DirectML), with automatic fallback to full precision if a GPU has trouble.
- **Speed-changed and reversed clips** are supported — stems are re-timed to match the clip on the timeline.
- **Nested sequences** are supported — Premiere renders the nested audio, then it is separated.
- Faster spectrogram processing, clearer errors, ZIP packages with uninstaller and README.

Website: https://framelabmedia.odoo.com · Instagram: [@bunnysframelab](https://www.instagram.com/bunnysframelab/)
