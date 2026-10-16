; Inno Setup script: wraps the PyInstaller output (dist\WhiteNoiseRemover) into one setup .exe.
; Build: iscc /DAppVersion=1.0.0 packaging\installer.iss   (run from the project root)
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "화이트 노이즈 제거 프로그램"

[Setup]
AppId={{6C1F2B7A-4E8D-4D3B-9A51-7B2E0C9F4A10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=hec110ys-maker
DefaultDirName={autopf}\WhiteNoiseRemover
DefaultGroupName={#AppName}
; Per-user install: no administrator password needed
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=WhiteNoiseRemover-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\WhiteNoiseRemover.exe
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 작업:"

[Files]
Source: "..\dist\WhiteNoiseRemover\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\WhiteNoiseRemover.exe"
Name: "{group}\{#AppName} 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\WhiteNoiseRemover.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\WhiteNoiseRemover.exe"; Description: "지금 실행"; Flags: nowait postinstall skipifsilent
