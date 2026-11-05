@echo off
setlocal
title Uninstall Bunnys Audio Separator
set "EXT=%APPDATA%\Adobe\CEP\extensions\com.bunnysframelab.audioseparator"
set "DATA=%LOCALAPPDATA%\BunnysFramelab\AudioSeparator"

echo.
echo   Bunnys Audio Separator - Uninstall
echo   ----------------------------------
echo   This removes the plugin and its downloaded AI models.
echo   Please close Adobe Premiere Pro first.
echo.
pause

if exist "%EXT%\unins000.exe" (
  echo Running the uninstaller...
  "%EXT%\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
  timeout /t 4 /nobreak >nul
)
if exist "%EXT%" rmdir /s /q "%EXT%"
if exist "%DATA%" rmdir /s /q "%DATA%"

echo.
if exist "%EXT%" (
  echo   Some files could not be removed. Close Premiere Pro and run this again.
) else (
  echo   Done. Bunnys Audio Separator has been removed.
)
echo.
pause
