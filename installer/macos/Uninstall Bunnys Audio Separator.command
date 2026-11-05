#!/bin/bash
# Bunnys Audio Separator - Uninstall (macOS)
EXT="/Library/Application Support/Adobe/CEP/extensions/com.bunnysframelab.audioseparator"
DATA="$HOME/Library/Application Support/BunnysFramelab/AudioSeparator"

echo ""
echo "  Bunnys Audio Separator - Uninstall"
echo "  ----------------------------------"
echo "  This removes the plugin and its downloaded AI models."
echo "  Please quit Adobe Premiere Pro first."
echo "  You will be asked for your Mac password."
echo ""
read -r -p "  Press Enter to continue (or Ctrl+C to cancel)... "

sudo rm -rf "$EXT"
rm -rf "$DATA"
sudo pkgutil --forget com.bunnysframelab.audioseparator >/dev/null 2>&1

echo ""
if [ -d "$EXT" ]; then
  echo "  Some files could not be removed. Quit Premiere Pro and run this again."
else
  echo "  Done. Bunnys Audio Separator has been removed."
fi
echo ""
