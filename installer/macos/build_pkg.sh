#!/bin/bash
# Build the macOS installer package from dist/extension (run on macOS).
#   installer/macos/build_pkg.sh <version> <arch-label>
set -euo pipefail
VERSION="${1:-0.1.0}"
ARCH="${2:-$(uname -m)}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUNDLE="com.bunnysframelab.audioseparator"
PAYLOAD="$ROOT/build/pkgroot"
DEST="$PAYLOAD/Library/Application Support/Adobe/CEP/extensions/$BUNDLE"

rm -rf "$PAYLOAD"
mkdir -p "$(dirname "$DEST")"
cp -R "$ROOT/dist/extension/$BUNDLE" "$DEST"
chmod +x "$DEST/bin/bunnys-engine/bunnys-engine"
chmod +x "$ROOT/installer/macos/scripts/"*

pkgbuild \
  --root "$PAYLOAD" \
  --install-location / \
  --identifier "$BUNDLE" \
  --version "$VERSION" \
  --scripts "$ROOT/installer/macos/scripts" \
  "$ROOT/dist/BunnysAudioSeparator-macOS-$ARCH-v$VERSION.pkg"

echo "built dist/BunnysAudioSeparator-macOS-$ARCH-v$VERSION.pkg"
