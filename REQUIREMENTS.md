# Bunnys Audio Separator — requirements (from user, 2026-11-04 00:38 Asia/Dhaka)

- Name: "Bunnys Audio Separator", created by Bunnys Framelab
- Host: Adobe Premiere Pro panel (UXP, Premiere 2025+), Windows + macOS
- Theme: matches framelabmedia.odoo.com
  - Primary: Electric Blue #1F6BFF
  - Accent: Sunset Orange #FF8A00
  - Secondary/background: Deep Black #08080A
  - Gradient mids (blue -> orange path, as in logo): Violet #6A2BFF, Hot Red #FF1F56
- Logo: assets/logo.webp (angry bunny face, blue->violet->red->orange gradient on black)

## Modes
1. Music stems: Vocals, Drums, Bass, Guitar, Piano, Other — checkboxes, user picks which stems to add
   (Demucs htdemucs_6s; vocal/music 2-stem shortcut)
2. Dialogue/Vocal, SFX, Music separation (cinematic source separation, e.g. DnR-style model)

## Workflow
- User selects an audio clip/track in timeline, clicks button -> separation runs -> selected stems
  imported and placed on new audio tracks, synced to original clip

## Performance
- Device option: Auto / GPU / CPU
- Auto picks best backend: CUDA/TensorRT or DirectML (Windows), CoreML (Apple Silicon), CPU fallback
- As lightweight, fast and smooth as possible while keeping quality; runs on all kinds of PCs
  (FP16, VRAM-aware chunking, ONNX Runtime, no Python needed for end users)

## Footer
- Links: website framelabmedia.odoo.com, Instagram @bunnysframelab

## Delivery
- Code in GitHub repo, GitHub Actions builds Windows .exe + macOS .pkg + panel .ccx into Releases
- Free distribution, unsigned installers for v0.1 (README explains warnings)
