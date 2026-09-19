; Inno Setup script for CoderAI-Setup.exe — installs the Windows launcher and,
; on first launch, runs install-coderai.ps1 (WSL2, Docker Desktop, the image).
; Build: iscc CoderAI.iss   (or packaging/windows/build-installer.sh under wine)
#ifndef AppVersion
  #define AppVersion "0.2.20"
#endif
[Setup]
AppId={{7C1E7F2A-6B3D-4B2D-9C7B-CODERAI0001}
AppName=CoderAI
AppVersion={#AppVersion}
AppPublisher=Nexlab
AppPublisherURL=https://www.nexlab.net/projects/coderai/
AppSupportURL=https://aisbf.cloud/coderai/docs/
DefaultDirName={localappdata}\Programs\CoderAI
DefaultGroupName=CoderAI
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=CoderAI-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
ChangesEnvironment=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "coderai.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "coderai.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "install-coderai.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\CoderAI"; Filename: "{app}\coderai.cmd"; WorkingDir: "{app}"
Name: "{group}\CoderAI — stop"; Filename: "{app}\coderai.cmd"; Parameters: "-Stop"; WorkingDir: "{app}"
Name: "{group}\CoderAI — upgrade"; Filename: "{app}\coderai.cmd"; Parameters: "-Upgrade"; WorkingDir: "{app}"
Name: "{group}\Set up WSL2 + Docker (admin)"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install-coderai.ps1"" -NoPull"; WorkingDir: "{app}"
Name: "{group}\CoderAI docs"; Filename: "https://aisbf.cloud/coderai/docs/install.html"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath('{app}')

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install-coderai.ps1"" -NoPull"; Description: "Set up WSL2 and Docker Desktop now (needs administrator)"; Flags: postinstall shellexec runascurrentuser skipifsilent; Verb: runas
Filename: "{app}\coderai.cmd"; Description: "Start CoderAI (pulls the ~28 GB image on first run)"; Flags: postinstall shellexec skipifsilent unchecked

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;
