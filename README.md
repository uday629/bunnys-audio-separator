<p align="center"><img src="panel/assets/logo.png" width="110" alt="Bunnys Framelab logo"></p>

<h1 align="center">Bunnys Audio Separator</h1>
<p align="center">AI stem separation inside Adobe Premiere Pro — created by <b>Bunnys Framelab</b><br>
<a href="https://framelabmedia.odoo.com">framelabmedia.odoo.com</a> · <a href="https://www.instagram.com/bunnysframelab/">Instagram @bunnysframelab</a></p>

---

## বাংলা

### এটা কী করে
Premiere Pro-র timeline-এ একটা audio clip select করে একটা button চাপলেই গানটা আলাদা আলাদা track-এ ভাগ হয়ে যায়:

- **Music Stems:** Vocals, Drums, Bass, Guitar, Piano, Other। সাথে আছে **Instrumental**, মানে vocal ছাড়া বাকি সব একসাথে। যেগুলো দরকার শুধু সেগুলো বাছতে পারবেন।
- **Vocal · SFX · Music (Beta):** ভিডিও, vlog বা ফিল্মের audio থেকে কথা, sound effect আর music আলাদা করে।

আলাদা হওয়া প্রতিটা stem নতুন audio track-এ মূল clip-এর সাথে একদম sync হয়ে বসে।

### Install
1. [Releases](../../releases/latest) page থেকে আপনার computer-এর file নামান:
   - Windows: `BunnysAudioSeparator-Windows-….exe`
   - Mac (M1/M2/M3/M4): `…macOS-AppleSilicon….pkg`
   - Mac (Intel): `…macOS-Intel….pkg`
2. Installer চালান।
   - **Windows:** "Windows protected your PC" দেখালে **More info → Run anyway** চাপুন।
   - **Mac:** block করলে **System Settings → Privacy & Security** থেকে **Open Anyway** চাপুন।
   - এই সতর্কবার্তা আসে কারণ installer এখনো code-sign করা হয়নি।
3. Premiere Pro restart করুন, তারপর **Window → Extensions → Bunnys Audio Separator** খুলুন।
4. প্রথমবার separate করার সময় AI model (~১১০ MB) নিজে থেকে download হবে। এরপর আর internet লাগবে না।

### ব্যবহার
1. Timeline-এ একটা audio clip (বা audio-সহ video clip) select করুন।
2. Mode বাছুন, তারপর যে stem-গুলো দরকার সেগুলোতে tick দিন।
3. **Device** বাছুন:
   - **Auto:** সবচেয়ে ভালো GPU নিজে খুঁজে নেয় (recommended)
   - **GPU:** জোর করে GPU-তে চালায়
   - **CPU:** GPU ছাড়া চালায়
4. **Quality** বাছুন:
   - **Fast:** সবচেয়ে দ্রুত
   - **Balanced:** দ্রুত আর ভালো মানের মাঝামাঝি (recommended)
   - **Best:** সবচেয়ে ভালো মান, তবে সময় প্রায় দ্বিগুণ লাগে
5. **Separate selected clip** চাপুন।

Stem-গুলো সেভ হয় project file-এর পাশে `Bunnys Stems` folder-এ। Premiere-এর ভেতরেও **Bunnys Stems** নামে একটা bin-এ পাবেন।

### GPU support
| Computer | যেভাবে চলবে |
|---|---|
| Windows + যেকোনো DirectX 12 GPU (NVIDIA / AMD / Intel) | DirectML দিয়ে GPU acceleration |
| Mac Apple Silicon | CoreML (GPU + Neural Engine) |
| GPU নেই এমন PC বা Intel Mac | CPU-তে চলবে, তবে ধীরে |

কোনো কারণে GPU চালু না হলে বা GPU-র memory কম পড়লে plugin নিজে থেকে ছোট batch-এ বা CPU-তে চলে যায়। কাজ মাঝপথে থেমে যায় না।

---

## English

### What it does
Select an audio clip in your Premiere Pro timeline, click one button, and get separated stems on new, perfectly synced audio tracks:

- **Music Stems:** Vocals, Drums, Bass, Guitar, Piano, Other, plus **Instrumental** (everything except vocals). Pick only what you need.
- **Vocal · SFX · Music (Beta):** splits dialogue, sound effects and music in video audio.

### Install
1. Download the installer for your computer from [Releases](../../releases/latest).
2. Run it. The installers are **not code-signed yet**:
   - **Windows:** if SmartScreen appears, click **More info → Run anyway**.
   - **macOS:** if the package is blocked, use **System Settings → Privacy & Security → Open Anyway**.
3. Restart Premiere Pro and open **Window → Extensions → Bunnys Audio Separator**.
4. The AI model (~110 MB) downloads automatically on first use.

### How it works
```
Premiere Pro panel (CEP, HTML/JS) ──spawns──► bunnys-engine (standalone, no Python needed)
        ▲                                         │ decode clip range (FFmpeg/PyAV)
        │ ExtendScript: import + place stems      │ HTDemucs 6-stem model on ONNX Runtime
        └──────────── JSON progress lines ◄───────┘ (CUDA / DirectML / CoreML / CPU)
```
- **Engine** (`engine/`): the neural network runs in ONNX Runtime. The STFT/iSTFT parts run in NumPy, so the same model file works on every GPU backend. Long audio is processed in overlapping segments with a cross-fade, identical to Demucs `apply_model`. CI verifies that the output matches PyTorch Demucs.
- **Only the used part of the clip is processed** (its in/out range), not the whole source file.
- **Automatic fallback:** on low GPU memory the batch is halved, and if the GPU fails the engine continues on the CPU.
- **Panel** (`panel/`): HTML/CSS/JS with ExtendScript (`jsx/host.jsx`) for reading the selection, importing stems into a bin and placing them on free or new audio tracks.

### Build from source
```bash
pip install -r engine/requirements-windows.txt   # or requirements-macos.txt
python engine/build_engine.py                     # -> dist/extension/<bundle>/
```
CI (`.github/workflows/`):
- `models.yml` exports the ONNX model and publishes it to the `models-v1` release.
- `release.yml` builds and smoke-tests the Windows `.exe` and macOS `.pkg`, then publishes the release.

### Known limitations (v0.1)
- Tested automatically in CI on Windows and macOS runners, without a GPU and without Premiere Pro. **Needs real testing inside Premiere Pro**, especially placing stems on tracks.
- Vocal · SFX · Music is derived from the music model (dialogue = vocals, SFX = other, music = drums + bass + guitar + piano). A dedicated dialogue/effects model is planned.
- Guitar and piano stems are less clean than vocals, drums and bass (a limitation of the current model).
- No fp16 model yet. NVIDIA GPUs run via DirectML, not CUDA.
- Clips with speed changes, and nested sequences, are not supported yet.

Third-party licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
