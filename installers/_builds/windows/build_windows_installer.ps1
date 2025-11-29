#!/usr/bin/env pwsh
param(
	[string]$Version,
	[switch]$KeepBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
	param([string]$Message)
	Write-Host "[i] $Message"
}

function Write-Warn {
	param([string]$Message)
	Write-Warning $Message
}

function Write-ErrorAndExit {
	param([string]$Message)
	Write-Error $Message
	exit 1
}

function Remove-PathRobust {
	param([string]$Path)
	if (-not $Path) { return }
	if (-not (Test-Path -LiteralPath $Path)) { return }
	try {
		Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
		return
	} catch {
		# Fallback to extended path removal via cmd.exe
	}

	$extended = if ($Path.StartsWith('\\?\')) { $Path } else { "\\\\?\$Path" }
	$escaped = $extended.Replace('"', '""')
	$isDirectory = [System.IO.Directory]::Exists($extended)
	$command = if ($isDirectory) {
		"rd /s /q `"$escaped`""
	} else {
		"del /f /q `"$escaped`""
	}

	$proc = Start-Process -FilePath cmd.exe -ArgumentList @('/c', $command) -NoNewWindow -Wait -PassThru
	if ($proc.ExitCode -ne 0 -and (Test-Path -LiteralPath $Path)) {
		Write-Warn "Failed to remove $Path (exit code $($proc.ExitCode))"
	}
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..\..")).Path
$buildRoot = Join-Path $scriptRoot "build"
$stagingDir = Join-Path $buildRoot "staging"
$issPath = Join-Path $buildRoot "ai-os-installer.iss"
$releaseDir = Join-Path $repoRoot "installers\releases"

if (-not $Version) {
	$pyprojectPath = Join-Path $repoRoot "pyproject.toml"
	if (-not (Test-Path $pyprojectPath)) {
		Write-ErrorAndExit "Unable to locate pyproject.toml at $pyprojectPath"
	}

	$versionMatch = Select-String -Path $pyprojectPath -Pattern '^[ \t]*version[ \t]*=[ \t]*"([^"]+)"' -Encoding UTF8 |
		Select-Object -First 1
	if (-not $versionMatch) {
		Write-ErrorAndExit "Unable to determine version from pyproject.toml"
	}
	$Version = $versionMatch.Matches[0].Groups[1].Value
}

if (-not $Version) {
	Write-ErrorAndExit "Version value is empty"
}

Write-Info "Preparing AI-OS Windows installer build (version $Version)"

if (Test-Path $buildRoot) {
	Remove-Item -Path $buildRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

$projectItems = @(
	'artifacts',
	'config',
	'docs',
	'installers',
	'src',
	'training_data',
	'training_datasets',
	'logging.yaml',
	'pyproject.toml',
	'README.md',
	'LICENSE',
	'NOTICE',
	'ruff.toml',
	'checklist.txt',
	'launcher.bat'
)

foreach ($item in $projectItems) {
	$sourcePath = Join-Path $repoRoot $item
	if (-not (Test-Path $sourcePath)) {
		Write-Info "Skipping missing item: $item"
		continue
	}

	$destinationPath = Join-Path $stagingDir $item
	$sourceInfo = Get-Item $sourcePath
	if ($sourceInfo.PSIsContainer) {
		Write-Info "Copying directory $item"
		New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
		$excludeNames = @()
		if ($item -eq 'installers') {
			$excludeNames = @('_builds', 'releases')
		}

		Get-ChildItem -Path $sourcePath -Force | ForEach-Object {
			if ($excludeNames -and $_.PSIsContainer -and ($excludeNames -contains $_.Name)) {
				Write-Info "Skipping excluded subdirectory $item/$($_.Name)"
				return
			}
			$target = Join-Path $destinationPath $_.Name
			Copy-Item -Path $_.FullName -Destination $target -Recurse -Force
		}
	} else {
		Write-Info "Copying file $item"
		$destinationParent = Split-Path -Parent $destinationPath
		if ($destinationParent -and -not (Test-Path $destinationParent)) {
			New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
		}
		Copy-Item -Path $sourcePath -Destination $destinationPath -Force
	}
}

$prunePaths = @(
	"$stagingDir\installers\_builds",
	"$stagingDir\installers\releases",
	"$stagingDir\logs",
	"$stagingDir\.git",
	"$stagingDir\.github",
	"$stagingDir\.venv",
	"$stagingDir\artifacts\evaluation"
)

foreach ($path in $prunePaths) {
	if (Test-Path -LiteralPath $path) {
		Write-Info "Pruning staged path $path"
		Remove-PathRobust -Path $path
	}
}

$brainsDir = Join-Path $stagingDir "artifacts\brains"
if (-not (Test-Path -LiteralPath $brainsDir)) {
	New-Item -ItemType Directory -Path $brainsDir -Force | Out-Null
}
if (Test-Path -LiteralPath $brainsDir) {
	Write-Info "Removing bundled brain artifacts from $brainsDir"
	Get-ChildItem -Path $brainsDir -Directory -Force | ForEach-Object {
		Remove-PathRobust -Path $_.FullName
	}
	Get-ChildItem -Path $brainsDir -File -Force | Where-Object { $_.Name -notin @('masters.json','pinned.json') } | ForEach-Object {
		Remove-PathRobust -Path $_.FullName
	}
	@('masters.json','pinned.json') | ForEach-Object {
		$target = Join-Path $brainsDir $_
		Set-Content -Path $target -Value '[]' -Encoding ASCII
	}
}

$stagingDir = (Resolve-Path $stagingDir).Path
$repoRoot = (Resolve-Path $repoRoot).Path
$releaseDir = (Resolve-Path $releaseDir).Path

$payloadBytes = (Get-ChildItem -LiteralPath $stagingDir -Recurse -File -ErrorAction SilentlyContinue |
	Measure-Object -Property Length -Sum).Sum
if (-not $payloadBytes) { $payloadBytes = 0 }
$payloadBytes = [long]$payloadBytes

function Find-InnoSetupCompiler {
	$candidates = @()
	try {
		$cmd = Get-Command ISCC.exe -ErrorAction Stop
		if ($cmd -and $cmd.Source) {
			$candidates += $cmd.Source
		}
	} catch {
		# Ignore lookup failures; fall back to known locations.
	}

	$programFiles = [Environment]::GetEnvironmentVariable('ProgramFiles')
	if ($programFiles -and (Test-Path $programFiles)) {
		$candidates += Join-Path $programFiles 'Inno Setup 6\ISCC.exe'
	}
	$programFilesX86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
	if ($programFilesX86 -and (Test-Path $programFilesX86)) {
		$candidates += Join-Path $programFilesX86 'Inno Setup 6\ISCC.exe'
	}

	$localAppData = [Environment]::GetEnvironmentVariable('LocalAppData')
	if ($localAppData -and (Test-Path $localAppData)) {
		$candidates += Join-Path $localAppData 'Programs\Inno Setup 6\ISCC.exe'
	}

	foreach ($candidate in $candidates | Select-Object -Unique) {
		if ($candidate -and (Test-Path $candidate)) {
			return (Resolve-Path $candidate).Path
		}
	}

	Write-ErrorAndExit "Inno Setup compiler (ISCC.exe) not found. Install Inno Setup 6 and ensure ISCC.exe is in PATH."
}

$isccPath = Find-InnoSetupCompiler
Write-Info "Using Inno Setup compiler at $isccPath"

$appId = '{{C0A4D2C3-4E4C-4E6F-9E52-BAA0C9A0F3F3}}'

$issContent = @"
#define MyAppName "AI-OS"
#define MyAppVersion "$Version"
#define MyAppPublisher "Wulfic"
#define MyAppURL "https://github.com/Wulfic/AI-OS"
#define SourceRoot "$stagingDir"
#define OutputRoot "$releaseDir"
#define RepoRoot "$repoRoot"
#define CorePayloadBytes $payloadBytes

[Setup]
AppId=$appId
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={autopf}\AI-OS
DefaultGroupName=AI-OS
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\installers\AI-OS.ico
OutputDir={#OutputRoot}
OutputBaseFilename=AI-OS-{#MyAppVersion}-Setup
SetupIconFile={#RepoRoot}\installers\AI-OS.ico
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
ChangesEnvironment=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#SourceRoot}\installers\scripts\install_aios_on_windows.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "{#SourceRoot}\LICENSE"; DestDir: "{tmp}"; Flags: dontcopy
Source: "{#SourceRoot}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Run]
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoLogo -NonInteractive -File ""{app}\installers\scripts\install_aios_on_windows.ps1"" -Action install -Yes"; \
  Description: "Configure AI-OS runtime (requires internet access)"; \
  StatusMsg: "Configuring AI-OS runtime environment..."; \
  Flags: waituntilterminated

[UninstallRun]
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoLogo -NonInteractive -File ""{app}\installers\scripts\install_aios_on_windows.ps1"" -Action uninstall -Yes"; \
  RunOnceId: "AIOSUninstall"; \
  Flags: waituntilterminated
"@

$codeBlock = @'
[Code]
const
	PreflightPayloadBytes = {#CorePayloadBytes};
	PreflightScriptName = 'install_aios_on_windows.ps1';
	PreflightOutputName = 'aios_preflight.txt';
	LicenseFileName = 'LICENSE';

var
	LicensePage: TWizardPage;
	LicenseViewer: TRichEditViewer;
	LicenseAcceptCheck: TNewCheckBox;
	DiskSummaryLabel: TNewStaticText;
	DiskDetailLabel: TNewStaticText;
	DiskRetryButton: TNewButton;
	PreflightOk: Boolean;
	PreflightRunning: Boolean;
	PreflightCoreBytes: Int64;
	PreflightDepBytes: Int64;
	PreflightBufferBytes: Int64;
	PreflightTotalBytes: Int64;
	PreflightMode: string;
	PreflightNote: string;

procedure StartPreflight(const ShowErrors: Boolean); forward;

function FormatBytes(Value: Int64): string;
begin
	if Value <= 0 then
		Result := '0 B'
	else if Value >= 1073741824 then
		Result := Format('%.2f GB', [Value / 1073741824.0])
	else if Value >= 1048576 then
		Result := Format('%.1f MB', [Value / 1048576.0])
	else
		Result := Format('%d KB', [Value div 1024]);
end;

function GetPowerShellPath: string;
begin
	Result := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
	if not FileExists(Result) then
		Result := 'powershell.exe';
end;

procedure UpdateNextButtonState;
begin
	if (LicensePage = nil) or (WizardForm = nil) then
		Exit;
	if WizardForm.CurPageID = LicensePage.ID then
		WizardForm.NextButton.Enabled :=
			LicenseAcceptCheck.Checked and PreflightOk and (not PreflightRunning);
end;

procedure LicenseAcceptChanged(Sender: TObject);
begin
	UpdateNextButtonState;
end;

procedure SetDiskStatus(const Summary, Detail: string; ErrorState: Boolean);
begin
	DiskSummaryLabel.Caption := Summary;
	DiskDetailLabel.Caption := Detail;
	if ErrorState then
		DiskSummaryLabel.Font.Color := clRed
	else
		DiskSummaryLabel.Font.Color := clWindowText;
end;

procedure ResetPreflightValues;
begin
	PreflightCoreBytes := 0;
	PreflightDepBytes := 0;
	PreflightBufferBytes := 0;
	PreflightTotalBytes := 0;
	PreflightMode := '';
	PreflightNote := '';
end;

function LoadPreflightOutput(const FileName: string): Boolean;
var
	Lines: TStringList;
	I, P: Integer;
	Key, Value: string;
begin
	Result := False;
	if not FileExists(FileName) then
		Exit;
	ResetPreflightValues;
	Lines := TStringList.Create;
	try
		Lines.LoadFromFile(FileName);
		for I := 0 to Lines.Count - 1 do
		begin
			P := Pos('=', Lines[I]);
			if P <= 0 then
				Continue;
			Key := Trim(Copy(Lines[I], 1, P - 1));
			Value := Trim(Copy(Lines[I], P + 1, MaxInt));
			if Key = 'core_bytes' then
				PreflightCoreBytes := StrToInt64Def(Value, 0)
			else if Key = 'dependency_bytes' then
				PreflightDepBytes := StrToInt64Def(Value, 0)
			else if Key = 'buffer_bytes' then
				PreflightBufferBytes := StrToInt64Def(Value, 0)
			else if Key = 'total_bytes' then
				PreflightTotalBytes := StrToInt64Def(Value, 0)
			else if Key = 'mode' then
				PreflightMode := Value
			else if Key = 'note' then
				PreflightNote := Value;
		end;
		if PreflightTotalBytes = 0 then
			PreflightTotalBytes := PreflightCoreBytes + PreflightDepBytes + PreflightBufferBytes;
		Result := PreflightTotalBytes > 0;
	finally
		Lines.Free;
	end;
end;

procedure ShowPreflightSuccess;
var
	Detail: string;
begin
	Detail :=
		Format('Core files: %s  |  Dependencies: %s  |  Buffer: %s%sMode: %s%s%s',
			[FormatBytes(PreflightCoreBytes),
			 FormatBytes(PreflightDepBytes),
			 FormatBytes(PreflightBufferBytes),
			 #13#10,
			 PreflightMode,
			 #13#10,
			 PreflightNote]);
	SetDiskStatus(
		Format('Estimated total required: %s', [FormatBytes(PreflightTotalBytes)]),
		Detail,
		False);
end;

procedure StartPreflight(const ShowErrors: Boolean);
var
	PSExe, ScriptPath, OutputPath, Params: string;
	ResultCode: Integer;
begin
	PreflightRunning := True;
	PreflightOk := False;
	DiskRetryButton.Enabled := False;
	SetDiskStatus('Measuring disk usage...',
		'Gathering GPU + dependency footprint. This may take up to a minute.',
		False);
	UpdateNextButtonState;
	ExtractTemporaryFile(PreflightScriptName);
	ScriptPath := ExpandConstant('{tmp}\') + PreflightScriptName;
	OutputPath := ExpandConstant('{tmp}\') + PreflightOutputName;
	DeleteFile(OutputPath);
	PSExe := GetPowerShellPath;
	Params := Format('-ExecutionPolicy Bypass -NoLogo -NonInteractive -WindowStyle Hidden -File "%s" -Action preflight -Quiet -Gpu auto -PayloadBytes %d -PreflightOutput "%s"',
		[ScriptPath,
		 PreflightPayloadBytes,
		 OutputPath]);
	if not Exec(PSExe, Params, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
	begin
		SetDiskStatus('Unable to launch PowerShell preflight.',
			'Ensure Windows PowerShell is available, then click Recalculate.',
			True);
		if ShowErrors then
			MsgBox('PowerShell could not be launched to compute disk usage.', mbError, MB_OK);
	end
	else if ResultCode <> 0 then
	begin
		SetDiskStatus(
			Format('Preflight exited with code %d.', [ResultCode]),
			'Review installer.log for errors, then click Recalculate.',
			True);
		if ShowErrors then
			MsgBox('Disk usage estimation failed. Check your execution policy and retry.', mbError, MB_OK);
	end
	else if not LoadPreflightOutput(OutputPath) then
	begin
		SetDiskStatus('Preflight did not return usable data.',
			'The helper script finished without writing metrics. Click Recalculate to try again.',
			True);
		if ShowErrors then
			MsgBox('Disk usage estimation returned no data.', mbError, MB_OK);
	end
	else
	begin
		PreflightOk := True;
		ShowPreflightSuccess;
	end;
	PreflightRunning := False;
	DiskRetryButton.Enabled := True;
	UpdateNextButtonState;
end;

procedure DiskRetryButtonClick(Sender: TObject);
begin
	StartPreflight(True);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
	if (LicensePage <> nil) and (CurPageID <> LicensePage.ID) then
		WizardForm.NextButton.Enabled := True
	else
		UpdateNextButtonState;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
	Result := True;
	if (LicensePage <> nil) and (CurPageID = LicensePage.ID) then
	begin
		if not LicenseAcceptCheck.Checked then
		begin
			MsgBox('You must accept the AI-OS license agreement before continuing.', mbError, MB_OK);
			Result := False;
			Exit;
		end;
		if not PreflightOk then
		begin
			MsgBox('Wait for the disk usage estimate to finish (or click Recalculate) before continuing.', mbError, MB_OK);
			Result := False;
			Exit;
		end;
	end;
end;

procedure InitializeWizard;
var
	LicensePath: string;
begin
	LicensePage :=
		CreateCustomPage(
			wpWelcome,
			'License & Disk Usage',
			'Read the AI-OS terms and confirm there is enough free space before installing.');
	LicenseViewer := TRichEditViewer.Create(LicensePage.Surface);
	LicenseViewer.Parent := LicensePage.Surface;
	LicenseViewer.Left := 0;
	LicenseViewer.Top := 0;
	LicenseViewer.Width := LicensePage.SurfaceWidth;
	LicenseViewer.Height := ScaleY(150);
	LicenseViewer.ScrollBars := ssVertical;
	LicenseViewer.BorderStyle := bsSingle;
	LicenseViewer.ReadOnly := True;
	ExtractTemporaryFile(LicenseFileName);
	LicensePath := ExpandConstant('{tmp}\') + LicenseFileName;
	if FileExists(LicensePath) then
		LicenseViewer.Lines.LoadFromFile(LicensePath)
	else
		LicenseViewer.Lines.Text := 'License file missing from installer payload.';

	LicenseAcceptCheck := TNewCheckBox.Create(LicensePage.Surface);
	LicenseAcceptCheck.Parent := LicensePage.Surface;
	LicenseAcceptCheck.Caption := 'I accept the AI-OS License Agreement';
	LicenseAcceptCheck.Left := 0;
	LicenseAcceptCheck.Top := LicenseViewer.Top + LicenseViewer.Height + ScaleY(8);
	LicenseAcceptCheck.Width := LicensePage.SurfaceWidth;
	LicenseAcceptCheck.OnClick := @LicenseAcceptChanged;

	DiskSummaryLabel := TNewStaticText.Create(LicensePage.Surface);
	DiskSummaryLabel.Parent := LicensePage.Surface;
	DiskSummaryLabel.Left := 0;
	DiskSummaryLabel.Top := LicenseAcceptCheck.Top + LicenseAcceptCheck.Height + ScaleY(12);
	DiskSummaryLabel.Width := LicensePage.SurfaceWidth;
	DiskSummaryLabel.AutoSize := False;
	DiskSummaryLabel.Height := ScaleY(36);
	DiskSummaryLabel.WordWrap := True;
	DiskSummaryLabel.Font.Style := [fsBold];

	DiskDetailLabel := TNewStaticText.Create(LicensePage.Surface);
	DiskDetailLabel.Parent := LicensePage.Surface;
	DiskDetailLabel.Left := 0;
	DiskDetailLabel.Top := DiskSummaryLabel.Top + DiskSummaryLabel.Height + ScaleY(4);
	DiskDetailLabel.Width := LicensePage.SurfaceWidth;
	DiskDetailLabel.Height := ScaleY(60);
	DiskDetailLabel.AutoSize := False;
	DiskDetailLabel.WordWrap := True;

	DiskRetryButton := TNewButton.Create(LicensePage.Surface);
	DiskRetryButton.Parent := LicensePage.Surface;
	DiskRetryButton.Caption := '&Recalculate';
	DiskRetryButton.Width := ScaleX(110);
	DiskRetryButton.Left := 0;
	DiskRetryButton.Top := DiskDetailLabel.Top + DiskDetailLabel.Height + ScaleY(4);
	DiskRetryButton.OnClick := @DiskRetryButtonClick;

	StartPreflight(False);
	UpdateNextButtonState;
end;
'@

$issContent += $codeBlock

Set-Content -Path $issPath -Value $issContent -Encoding ASCII

Write-Info "Invoking Inno Setup compiler"
$process = Start-Process -FilePath $isccPath -ArgumentList "`"$issPath`"" -NoNewWindow -Wait -PassThru

if ($process.ExitCode -ne 0) {
	Write-ErrorAndExit "Inno Setup compiler failed with exit code $($process.ExitCode)"
}

$outputInstaller = Join-Path $releaseDir "AI-OS-$Version-Setup.exe"
if (-not (Test-Path $outputInstaller)) {
	Write-ErrorAndExit "Expected installer not found at $outputInstaller"
}

Write-Info "Created installer $outputInstaller"

if (-not $KeepBuild) {
	Write-Info "Cleaning build workspace"
	Remove-Item -Path $buildRoot -Recurse -Force
}

Write-Info "Done"