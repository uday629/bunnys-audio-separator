; Bunnys Audio Separator — Windows installer (Inno Setup 6)
; Built by CI:  iscc /DAppVersion=0.1.0 installer\windows\setup.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define BundleId "com.bunnysframelab.audioseparator"

[Setup]
AppId={{8E5B2C1A-6F0D-4D8B-9B7E-B0BB1E5A0D10}
AppName=Bunnys Audio Separator
AppVersion={#AppVersion}
AppPublisher=Bunnys Framelab
AppPublisherURL=https://framelabmedia.odoo.com
AppSupportURL=https://framelabmedia.odoo.com
DefaultDirName={userappdata}\Adobe\CEP\extensions\{#BundleId}
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=BunnysAudioSeparator-Windows-v{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\assets\icon.png
UninstallDisplayName=Bunnys Audio Separator
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\..\dist\extension\{#BundleId}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[InstallDelete]
; remove files from an older version before copying the new one
Type: filesandordirs; Name: "{app}\bin"

[Registry]
; Allow this (self-distributed) panel to load in every recent Premiere Pro version
Root: HKCU; Subkey: "Software\Adobe\CSXS.9";  ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"
Root: HKCU; Subkey: "Software\Adobe\CSXS.10"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"
Root: HKCU; Subkey: "Software\Adobe\CSXS.11"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"
Root: HKCU; Subkey: "Software\Adobe\CSXS.12"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"
Root: HKCU; Subkey: "Software\Adobe\CSXS.13"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{localappdata}\BunnysFramelab\AudioSeparator"

[Messages]
FinishedLabel=Bunnys Audio Separator is installed.%n%nRestart Premiere Pro, then open:%nWindow > Extensions > Bunnys Audio Separator
