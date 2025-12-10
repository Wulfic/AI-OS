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

	# For directories, prefer cmd.exe /c rd /s /q as it is faster and handles deep paths/locks better
	if (Test-Path -LiteralPath $Path -PathType Container) {
		$extended = if ($Path.StartsWith('\\?\')) { $Path } else { "\\\\?\$Path" }
		$escaped = $extended.Replace('"', '""')
		$command = "rd /s /q `"$escaped`""
		
		$proc = Start-Process -FilePath cmd.exe -ArgumentList @('/c', $command) -NoNewWindow -Wait -PassThru
		if ($proc.ExitCode -eq 0 -and -not (Test-Path -LiteralPath $Path)) {
			return
		}
		# If cmd failed, fall through to Remove-Item as backup
	}

	try {
		$ProgressPreference = 'SilentlyContinue'
		Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
		return
	} catch {
		# Fallback to extended path removal via cmd.exe (for files or if first attempt failed)
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

# Detect host Python version to embed in installer
$hostPythonVersion = "3.10.11"  # Default fallback
try {
	$pyVersionOutput = python --version 2>&1
	if ($LASTEXITCODE -eq 0 -and $pyVersionOutput -match 'Python\s+(\d+\.\d+\.\d+)') {
		$hostPythonVersion = $Matches[1]
		Write-Info "Detected host Python version: $hostPythonVersion"
	}
} catch {
	Write-Warn "Could not detect host Python version, using default: $hostPythonVersion"
}

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
			$excludeNames = @('_builds', 'releases', 'dependencies')
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
#define HostPythonVersion "$hostPythonVersion"

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
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
ChangesEnvironment=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#SourceRoot}\installers\scripts\install_aios_on_windows.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "{#SourceRoot}\LICENSE"; DestDir: "{tmp}"; Flags: dontcopy
Source: "{#SourceRoot}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

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
	PreflightLogName = 'aios_preflight.log';
	InstallLogName = 'aios_install.log';
	LicenseFileName = 'LICENSE';
	HostPythonVersion = '{#HostPythonVersion}';
	ProcessPollIntervalMs = 300;
	WAIT_OBJECT_0 = $00000000;
	WAIT_TIMEOUT = $00000102;
	WAIT_FAILED = $FFFFFFFF;
	SEE_MASK_NOCLOSEPROCESS = $00000040;
	PM_REMOVE = $0001;
	EM_SCROLLCARET = $00B7;
	WM_VSCROLL = $0115;
	SB_BOTTOM = 7;

type
	TShellExecuteInfo = record
		cbSize: DWORD;
		fMask: Cardinal;
		Wnd: HWND;
		lpVerb: string;
		lpFile: string;
		lpParameters: string;
		lpDirectory: string;
		nShow: Integer;
		hInstApp: THandle;
		lpIDList: DWORD;
		lpClass: string;
		hkeyClass: THandle;
		dwHotKey: DWORD;
		hMonitor: THandle;
		hProcess: THandle;
	end;
	TWinPoint = record
		X: Longint;
		Y: Longint;
	end;
	TMsg = record
		hwnd: HWND;
		message: UINT;
		wParam: Longint;
		lParam: Longint;
		time: DWORD;
		pt: TWinPoint;
	end;

var
	LicensePage: TWizardPage;
	LicenseViewer: TRichEditViewer;
	LicenseAcceptCheck: TNewCheckBox;
	
	ConfigPage: TWizardPage;
	BrainDownloadCheck: TNewCheckBox;
	AdvancedModeCheck: TNewCheckBox;
	ComponentsPanel: TPanel;
	CompPythonCheck: TNewCheckBox;
	CompGitCheck: TNewCheckBox;
	CompNodeCheck: TNewCheckBox;
	CompCudaCheck: TNewCheckBox;
	
	DiskSummaryLabel: TNewStaticText;
	DiskDetailLabel: TNewStaticText;
	DiskRetryButton: TNewButton;
	LogLabel: TNewStaticText;
	LogViewer: TRichEditViewer;
	
	InstallLogViewer: TRichEditViewer;
	InstallDetailsButton: TNewButton;
	InstallDetailsVisible: Boolean;
	InstallLogLastCount: Integer;
	
	PreflightOk: Boolean;
	PreflightRunning: Boolean;
	PreflightCoreBytes: Int64;
	PreflightDepBytes: Int64;
	PreflightBufferBytes: Int64;
	PreflightTotalBytes: Int64;
	PreflightMode: string;
	PreflightNote: string;
	PreflightPythonStatus: string;
	PreflightGitStatus: string;
	PreflightNodeStatus: string;

procedure UpdateConfigPageLayout; forward;

function ShellExecuteEx(var lpExecInfo: TShellExecuteInfo): BOOL;
	external 'ShellExecuteExW@shell32.dll stdcall';
function WaitForSingleObject(hHandle: THandle; dwMilliseconds: DWORD): DWORD;
	external 'WaitForSingleObject@kernel32.dll stdcall';
function GetExitCodeProcess(hProcess: THandle; var lpExitCode: DWORD): BOOL;
	external 'GetExitCodeProcess@kernel32.dll stdcall';
function CloseHandle(hObject: THandle): BOOL;
	external 'CloseHandle@kernel32.dll stdcall';
function GetLastError: DWORD;
	external 'GetLastError@kernel32.dll stdcall';
function PeekMessage(var lpMsg: TMsg; hWnd: HWND; wMsgFilterMin, wMsgFilterMax, wRemoveMsg: UINT): BOOL;
	external 'PeekMessageW@user32.dll stdcall';
function TranslateMessage(const lpMsg: TMsg): BOOL;
	external 'TranslateMessage@user32.dll stdcall';
function DispatchMessage(const lpMsg: TMsg): Longint;
	external 'DispatchMessageW@user32.dll stdcall';
function SendMessage(hWnd: HWND; Msg: UINT; wParam, lParam: Longint): Longint;
	external 'SendMessageW@user32.dll stdcall';
function MulDiv(nNumber, nNumerator, nDenominator: Integer): Integer;
	external 'MulDiv@kernel32.dll stdcall';

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
		WizardForm.NextButton.Enabled := LicenseAcceptCheck.Checked
	else if (ConfigPage <> nil) and (WizardForm.CurPageID = ConfigPage.ID) then
		WizardForm.NextButton.Enabled := PreflightOk and (not PreflightRunning);
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
	UpdateConfigPageLayout;
end;

procedure ResetPreflightValues;
begin
	PreflightCoreBytes := 0;
	PreflightDepBytes := 0;
	PreflightBufferBytes := 0;
	PreflightTotalBytes := 0;
	PreflightMode := '';
	PreflightNote := '';
	PreflightPythonStatus := '';
	PreflightGitStatus := '';
	PreflightNodeStatus := '';
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
				PreflightNote := Value
			else if Key = 'python_status' then
				PreflightPythonStatus := Value
			else if Key = 'git_status' then
				PreflightGitStatus := Value
			else if Key = 'node_status' then
				PreflightNodeStatus := Value;
		end;
		if PreflightTotalBytes = 0 then
			PreflightTotalBytes := PreflightCoreBytes + PreflightDepBytes + PreflightBufferBytes;
		Result := PreflightTotalBytes > 0;
	finally
		Lines.Free;
	end;
end;

procedure LoadPreflightLog(const FileName: string);
begin
	if LogViewer = nil then
		Exit;
	if FileExists(FileName) then
	begin
		try
			LogViewer.Lines.LoadFromFile(FileName);
			LogViewer.SelStart := Length(LogViewer.Lines.Text);
		except
			LogViewer.Lines.Text := 'Unable to read log output.';
		end;
	end
	else
		LogViewer.Lines.Text := 'Waiting for log output...';
end;

procedure ShowPreflightSuccess;
var
	Detail: string;
begin
	Detail :=
		Format('Core files: %s  |  Dependencies: %s  |  Buffer: %s%sMode: %s%s%s%sPrerequisites: Python [%s], Git [%s], Node [%s]', [
			FormatBytes(PreflightCoreBytes),
			FormatBytes(PreflightDepBytes),
			FormatBytes(PreflightBufferBytes),
			(#13#10),
			PreflightMode,
			(#13#10),
			PreflightNote,
			(#13#10),
			PreflightPythonStatus,
			PreflightGitStatus,
			PreflightNodeStatus]);
	SetDiskStatus(
		Format('Estimated total required: %s', [FormatBytes(PreflightTotalBytes)]),
		Detail,
		False);
end;

procedure CloseProcessHandle(var ProcessHandle: THandle);
begin
	if ProcessHandle <> 0 then
	begin
		CloseHandle(ProcessHandle);
		ProcessHandle := 0;
	end;
end;

function LaunchHiddenProcess(const FileName, Params: string; var ProcessHandle: THandle): Boolean;
var
	ExecInfo: TShellExecuteInfo;
begin
	ExecInfo.cbSize := SizeOf(ExecInfo);
	ExecInfo.fMask := SEE_MASK_NOCLOSEPROCESS;
	ExecInfo.Wnd := WizardForm.Handle;
	ExecInfo.lpVerb := 'open';
	ExecInfo.lpFile := FileName;
	ExecInfo.lpParameters := Params;
	ExecInfo.lpDirectory := '';
	ExecInfo.nShow := SW_HIDE;
	ExecInfo.hInstApp := 0;
	ExecInfo.lpIDList := 0;
	ExecInfo.lpClass := '';
	ExecInfo.hkeyClass := 0;
	ExecInfo.dwHotKey := 0;
	ExecInfo.hMonitor := 0;
	ExecInfo.hProcess := 0;
	Result := ShellExecuteEx(ExecInfo);
	if Result then
		ProcessHandle := ExecInfo.hProcess
	else
		ProcessHandle := 0;
end;

procedure ProcessMessages;
var
	Msg: TMsg;
begin
	while PeekMessage(Msg, 0, 0, 0, PM_REMOVE) do begin
		TranslateMessage(Msg);
		DispatchMessage(Msg);
	end;
end;

function WaitForProcessWithLog(ProcessHandle: THandle; const LogPath: string): DWORD;
var
	WaitResult: DWORD;
begin
	Result := WAIT_FAILED;
	if ProcessHandle = 0 then
		Exit;
	repeat
		LoadPreflightLog(LogPath);
		try
			ProcessMessages;
		except
		end;
		WaitResult := WaitForSingleObject(ProcessHandle, ProcessPollIntervalMs);
	until WaitResult <> WAIT_TIMEOUT;
	LoadPreflightLog(LogPath);
	Result := WaitResult;
end;

procedure StartPreflight(const ShowErrors: Boolean);
var
	PSExe, ScriptPath, OutputPath, LogPath, Params: string;
	ResultCode: Integer;
	ProcessHandle: THandle;
	WaitResult: DWORD;
	ExitCode: DWORD;
	LaunchError: DWORD;
begin
	PreflightRunning := True;
	PreflightOk := False;
	
	{ Disable controls during preflight }
	if CompPythonCheck <> nil then CompPythonCheck.Enabled := False;
	if CompGitCheck <> nil then CompGitCheck.Enabled := False;
	if CompNodeCheck <> nil then CompNodeCheck.Enabled := False;
	if CompCudaCheck <> nil then CompCudaCheck.Enabled := False;

	SetDiskStatus('Measuring disk usage...',
		'Gathering GPU + dependency footprint. This may take up to a minute.',
		False);
	if LogViewer <> nil then
		LogViewer.Lines.Text := 'Starting disk usage estimation...';
	UpdateNextButtonState;
	ExtractTemporaryFile(PreflightScriptName);
	ScriptPath := ExpandConstant('{tmp}\') + PreflightScriptName;
	OutputPath := ExpandConstant('{tmp}\') + PreflightOutputName;
	LogPath := ExpandConstant('{tmp}\') + PreflightLogName;
	DeleteFile(OutputPath);
	DeleteFile(LogPath);
	PSExe := GetPowerShellPath;
	Params := Format('-ExecutionPolicy Bypass -NoLogo -NonInteractive -WindowStyle Hidden -File "%s" -Action preflight -Quiet -Gpu auto -PayloadBytes %d -PreflightOutput "%s" -InstallerLog "%s"', [
		ScriptPath,
		PreflightPayloadBytes,
		OutputPath,
		LogPath]);

	if AdvancedModeCheck.Checked then
	begin
		if not CompPythonCheck.Checked then Params := Params + ' -SkipPythonInstall';
		if not CompGitCheck.Checked then Params := Params + ' -SkipGitInstall';
		if not CompNodeCheck.Checked then Params := Params + ' -SkipNodeInstall';
		if CompCudaCheck.Checked then Params := Params + ' -InstallCudaTools';
	end;
	ProcessHandle := 0;
	if not LaunchHiddenProcess(PSExe, Params, ProcessHandle) then
	begin
		LaunchError := GetLastError;
		SetDiskStatus('Unable to launch PowerShell preflight.',
			Format('Ensure Windows PowerShell is available (error %d).', [LaunchError]),
			True);
		if ShowErrors then
			MsgBox('PowerShell could not be launched to compute disk usage.', mbError, MB_OK);
	end
	else
	begin
		WaitResult := WaitForProcessWithLog(ProcessHandle, LogPath);
		if WaitResult = WAIT_FAILED then
		begin
			SetDiskStatus('Failed while monitoring PowerShell preflight.',
				'Setup could not read the helper log.',
				True);
			if ShowErrors then
				MsgBox('Failed while monitoring the PowerShell helper.', mbError, MB_OK);
		end
		else if WaitResult <> WAIT_OBJECT_0 then
		begin
			SetDiskStatus('PowerShell preflight exited unexpectedly.',
				'The helper returned an unexpected wait state.',
				True);
			if ShowErrors then
				MsgBox('PowerShell preflight exited unexpectedly.', mbError, MB_OK);
		end
		else
		begin
			if not GetExitCodeProcess(ProcessHandle, ExitCode) then
				ResultCode := 1
			else
				ResultCode := Integer(ExitCode);
			if ResultCode <> 0 then
			begin
				SetDiskStatus(
					Format('Preflight exited with code %d.', [ResultCode]),
					'Review installer.log for errors.',
					True);
				if ShowErrors then
					MsgBox('Disk usage estimation failed. Check your execution policy and retry.', mbError, MB_OK);
			end
			else if not LoadPreflightOutput(OutputPath) then
			begin
				SetDiskStatus('Preflight did not return usable data.',
					'The helper script finished without writing metrics.',
					True);
				if ShowErrors then
					MsgBox('Disk usage estimation returned no data.', mbError, MB_OK);
			end
			else
			begin
				PreflightOk := True;
				ShowPreflightSuccess;
			end;
		end;
	end;
	CloseProcessHandle(ProcessHandle);
	PreflightRunning := False;
	
	{ Re-enable controls }
	if CompPythonCheck <> nil then CompPythonCheck.Enabled := True;
	if CompGitCheck <> nil then CompGitCheck.Enabled := True;
	if CompNodeCheck <> nil then CompNodeCheck.Enabled := True;
	if CompCudaCheck <> nil then CompCudaCheck.Enabled := True;
	
	UpdateNextButtonState;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
	if (LicensePage <> nil) and (CurPageID = LicensePage.ID) then
		UpdateNextButtonState
	else if (ConfigPage <> nil) and (CurPageID = ConfigPage.ID) then
		UpdateNextButtonState
	else
		WizardForm.NextButton.Enabled := True;
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
	end;
	if (ConfigPage <> nil) and (CurPageID = ConfigPage.ID) then
	begin
		if not PreflightOk then
		begin
			MsgBox('Wait for the disk usage estimate to finish (or click Recalculate) before continuing.', mbError, MB_OK);
			Result := False;
			Exit;
		end;
	end;
end;

procedure InstallDetailsButtonClick(Sender: TObject);
begin
	InstallDetailsVisible := not InstallDetailsVisible;
	InstallLogViewer.Visible := InstallDetailsVisible;
	if InstallDetailsVisible then
		InstallDetailsButton.Caption := 'Hide Details ^'
	else
		InstallDetailsButton.Caption := 'Show Details v';
end;

procedure StartInstall;
var
	PSExe, ScriptPath, LogPath, Params: string;
	ProcessHandle: THandle;
	WaitResult: DWORD;
	ExitCode: DWORD;
	Lines: TStringList;
	LastLine: string;
begin
	WizardForm.StatusLabel.Caption := 'Configuring AI-OS runtime environment...';
	WizardForm.ProgressGauge.Style := npbstMarquee;
	
	ScriptPath := ExpandConstant('{app}\installers\scripts\install_aios_on_windows.ps1');
	LogPath := ExpandConstant('{tmp}\') + InstallLogName;
	DeleteFile(LogPath);
	
	PSExe := GetPowerShellPath;
	
	{ Build parameters - the script handles its own logging via -InstallerLog }
	{ Pass the host Python version so the installer uses the same version as the build machine }
	Params := Format('-ExecutionPolicy Bypass -NoLogo -NonInteractive -WindowStyle Hidden -File "%s" -Action install -Yes -InstallerLog "%s" -PreferredPythonVersion "%s"', [
		ScriptPath,
		LogPath,
		HostPythonVersion]);

	if BrainDownloadCheck.Checked then
		Params := Params + ' -DownloadBrain'
	else
		Params := Params + ' -SkipBrain';

	if AdvancedModeCheck.Checked then
	begin
		if not CompPythonCheck.Checked then Params := Params + ' -SkipPythonInstall';
		if not CompGitCheck.Checked then Params := Params + ' -SkipGitInstall';
		if not CompNodeCheck.Checked then Params := Params + ' -SkipNodeInstall';
		if CompCudaCheck.Checked then Params := Params + ' -InstallCudaTools';
	end;
		
	ProcessHandle := 0;
	if not LaunchHiddenProcess(PSExe, Params, ProcessHandle) then
	begin
		MsgBox('Failed to launch configuration script.', mbError, MB_OK);
		Exit;
	end;
	
	{ Initialize tracking variable }
	InstallLogLastCount := 0;
	
	Lines := TStringList.Create;
	try
		repeat
			if FileExists(LogPath) then
			begin
				try
					Lines.LoadFromFile(LogPath);
					if Lines.Count > 0 then
					begin
						LastLine := Lines[Lines.Count - 1];
						if Length(LastLine) > 80 then
							LastLine := Copy(LastLine, 1, 77) + '...';
						WizardForm.StatusLabel.Caption := LastLine;
						
						{ Only update the log viewer if there are new lines }
						if (InstallLogViewer <> nil) and (Lines.Count > InstallLogLastCount) then
						begin
							{ Update content }
							InstallLogViewer.Lines.Assign(Lines);
							InstallLogLastCount := Lines.Count;
							{ Force scroll to absolute bottom using WM_VSCROLL }
							SendMessage(InstallLogViewer.Handle, WM_VSCROLL, SB_BOTTOM, 0);
						end;
					end;
				except
					{ Ignore read errors }
				end;
			end;
			
			try
				ProcessMessages;
			except
			end;
			WaitResult := WaitForSingleObject(ProcessHandle, ProcessPollIntervalMs);
		until WaitResult <> WAIT_TIMEOUT;
	finally
		Lines.Free;
	end;
	
	if not GetExitCodeProcess(ProcessHandle, ExitCode) then
		ExitCode := 1;
		
	CloseProcessHandle(ProcessHandle);
	
	if ExitCode <> 0 then
	begin
		MsgBox(Format('Configuration script failed with exit code %d. Check %s for details.', [ExitCode, LogPath]), mbError, MB_OK);
	end;
	
	WizardForm.ProgressGauge.Style := npbstNormal;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
	if CurStep = ssPostInstall then
	begin
		StartInstall;
	end;
end;

procedure DeinitializeSetup();
var
	PreflightLogSrc, InstallLogSrc: string;
	PreflightLogDest, InstallLogDest: string;
	LogDir, DesktopDir, SourceDir: string;
	Timestamp: string;
	AppLogCopyOk: Boolean;
begin
	{ Copy log files to multiple locations for accessibility }
	Timestamp := GetDateTimeString('yyyymmdd_hhnnss', #0, #0);
	PreflightLogSrc := ExpandConstant('{tmp}\') + PreflightLogName;
	InstallLogSrc := ExpandConstant('{tmp}\') + InstallLogName;
	AppLogCopyOk := False;
	
	{ PRIORITY 1: Copy to source directory (releases folder where installer is located) }
	{ This is the most useful location for development/testing }
	try
		SourceDir := ExpandConstant('{src}');
		if DirExists(SourceDir) then
		begin
			if FileExists(PreflightLogSrc) then
			begin
				PreflightLogDest := SourceDir + '\AIOS_preflight_' + Timestamp + '.log';
				CopyFile(PreflightLogSrc, PreflightLogDest, False);
			end;
			
			if FileExists(InstallLogSrc) then
			begin
				InstallLogDest := SourceDir + '\AIOS_install_' + Timestamp + '.log';
				CopyFile(InstallLogSrc, InstallLogDest, False);
			end;
		end;
	except
		{ Source directory copy failed - continue to other locations }
	end;
	
	{ PRIORITY 2: Try to copy to installation directory's logs folder }
	try
		LogDir := ExpandConstant('{app}\logs');
		if not DirExists(LogDir) then
			ForceDirectories(LogDir);
		
		if DirExists(LogDir) then
		begin
			if FileExists(PreflightLogSrc) then
			begin
				PreflightLogDest := LogDir + '\aios_preflight_' + Timestamp + '.log';
				CopyFile(PreflightLogSrc, PreflightLogDest, False);
			end;
			
			if FileExists(InstallLogSrc) then
			begin
				InstallLogDest := LogDir + '\aios_install_' + Timestamp + '.log';
				CopyFile(InstallLogSrc, InstallLogDest, False);
			end;
			AppLogCopyOk := True;
		end;
	except
		{ App directory not available - will try desktop }
		AppLogCopyOk := False;
	end;
	
	{ PRIORITY 3: Copy to user desktop as a fallback }
	try
		DesktopDir := ExpandConstant('{userdesktop}');
		if DirExists(DesktopDir) then
		begin
			if FileExists(PreflightLogSrc) then
			begin
				PreflightLogDest := DesktopDir + '\AIOS_preflight_' + Timestamp + '.log';
				CopyFile(PreflightLogSrc, PreflightLogDest, False);
			end;
			
			if FileExists(InstallLogSrc) then
			begin
				InstallLogDest := DesktopDir + '\AIOS_install_' + Timestamp + '.log';
				CopyFile(InstallLogSrc, InstallLogDest, False);
			end;
		end;
	except
		{ Desktop copy failed }
	end;
end;

procedure ComponentChanged(Sender: TObject);
begin
	StartPreflight(False);
end;

procedure UpdateConfigPageLayout;
var
	TopOffset: Integer;
	RemainingHeight: Integer;
begin
	ComponentsPanel.Visible := AdvancedModeCheck.Checked;
	
	TopOffset := AdvancedModeCheck.Top + AdvancedModeCheck.Height + ScaleY(5);
	
	if AdvancedModeCheck.Checked then
	begin
		ComponentsPanel.Top := TopOffset;
		{ Recalculate panel height based on children to prevent cutoff }
		ComponentsPanel.Height := CompCudaCheck.Top + CompCudaCheck.Height + ScaleY(5);
		TopOffset := TopOffset + ComponentsPanel.Height + ScaleY(5);
	end;
	
	DiskSummaryLabel.Top := TopOffset;
	DiskSummaryLabel.Width := ConfigPage.SurfaceWidth;
	TopOffset := TopOffset + DiskSummaryLabel.Height + ScaleY(10);
	
	DiskDetailLabel.Top := TopOffset;
	DiskDetailLabel.Width := ConfigPage.SurfaceWidth;
	TopOffset := TopOffset + DiskDetailLabel.Height + ScaleY(10);
	
	LogLabel.Top := TopOffset;
	TopOffset := TopOffset + LogLabel.Height + ScaleY(5);
	
	LogViewer.Top := TopOffset;
	RemainingHeight := ConfigPage.SurfaceHeight - TopOffset;
	if RemainingHeight < ScaleY(50) then RemainingHeight := ScaleY(50);
	LogViewer.Height := RemainingHeight;
end;

procedure AdvancedModeChanged(Sender: TObject);
begin
	UpdateConfigPageLayout;
end;

procedure InitializeWizard;
var
	LicensePath: string;
begin
	{ --- Page 1: License --- }
	LicensePage :=
		CreateCustomPage(
			wpWelcome,
			'License Agreement',
			'Please read the following important information before continuing.');
			
	LicenseAcceptCheck := TNewCheckBox.Create(LicensePage.Surface);
	LicenseAcceptCheck.Parent := LicensePage.Surface;
	LicenseAcceptCheck.Caption := 'I accept the AI-OS License Agreement';
	LicenseAcceptCheck.Left := 0;
	LicenseAcceptCheck.Height := ScaleY(20);
	LicenseAcceptCheck.Top := LicensePage.SurfaceHeight - LicenseAcceptCheck.Height - ScaleY(5);
	LicenseAcceptCheck.Width := LicensePage.SurfaceWidth;
	LicenseAcceptCheck.OnClick := @LicenseAcceptChanged;

	LicenseViewer := TRichEditViewer.Create(LicensePage.Surface);
	LicenseViewer.Parent := LicensePage.Surface;
	LicenseViewer.Left := 0;
	LicenseViewer.Top := 0;
	{ Extend to ~85% of page width to use more space }
	LicenseViewer.Width := MulDiv(WizardForm.ClientWidth, 85, 100);
	LicenseViewer.Height := LicenseAcceptCheck.Top - ScaleY(10);
	LicenseViewer.ScrollBars := ssVertical;
	LicenseViewer.BorderStyle := bsSingle;
	LicenseViewer.ReadOnly := True;
	ExtractTemporaryFile(LicenseFileName);
	LicensePath := ExpandConstant('{tmp}\') + LicenseFileName;
	if FileExists(LicensePath) then
		LicenseViewer.Lines.LoadFromFile(LicensePath)
	else
		LicenseViewer.Lines.Text := 'License file missing from installer payload.';

	{ --- Page 2: Configuration --- }
	ConfigPage :=
		CreateCustomPage(
			LicensePage.ID,
			'Installation Configuration',
			'Select components and verify disk usage.');

	BrainDownloadCheck := TNewCheckBox.Create(ConfigPage.Surface);
	BrainDownloadCheck.Parent := ConfigPage.Surface;
	BrainDownloadCheck.Caption := 'Download pretrained brain (English-v1) [~2GB]';
	BrainDownloadCheck.Left := 0;
	BrainDownloadCheck.Top := 0;
	BrainDownloadCheck.Width := ConfigPage.SurfaceWidth;
	BrainDownloadCheck.Height := ScaleY(30);
	BrainDownloadCheck.Checked := True;

	AdvancedModeCheck := TNewCheckBox.Create(ConfigPage.Surface);
	AdvancedModeCheck.Parent := ConfigPage.Surface;
	AdvancedModeCheck.Caption := 'Advanced Installation Mode (Select specific components)';
	AdvancedModeCheck.Left := 0;
	AdvancedModeCheck.Top := BrainDownloadCheck.Top + BrainDownloadCheck.Height + ScaleY(5);
	AdvancedModeCheck.Width := ConfigPage.SurfaceWidth;
	AdvancedModeCheck.Height := ScaleY(20);
	AdvancedModeCheck.Checked := False;
	AdvancedModeCheck.OnClick := @AdvancedModeChanged;

	ComponentsPanel := TPanel.Create(ConfigPage.Surface);
	ComponentsPanel.Parent := ConfigPage.Surface;
	ComponentsPanel.Left := ScaleX(20);
	ComponentsPanel.Width := ConfigPage.SurfaceWidth - ScaleX(20);
	ComponentsPanel.Height := ScaleY(100);
	ComponentsPanel.BevelOuter := bvNone;
	ComponentsPanel.Visible := False;

	CompPythonCheck := TNewCheckBox.Create(ComponentsPanel);
	CompPythonCheck.Parent := ComponentsPanel;
	CompPythonCheck.Caption := 'Install Python 3.11 (if missing)';
	CompPythonCheck.Left := 0;
	CompPythonCheck.Top := 0;
	CompPythonCheck.Width := ComponentsPanel.Width;
	CompPythonCheck.Height := ScaleY(20);
	CompPythonCheck.Checked := True;
	CompPythonCheck.OnClick := @ComponentChanged;

	CompGitCheck := TNewCheckBox.Create(ComponentsPanel);
	CompGitCheck.Parent := ComponentsPanel;
	CompGitCheck.Caption := 'Install Git (if missing)';
	CompGitCheck.Left := 0;
	CompGitCheck.Top := CompPythonCheck.Top + CompPythonCheck.Height + ScaleY(4);
	CompGitCheck.Width := ComponentsPanel.Width;
	CompGitCheck.Height := ScaleY(20);
	CompGitCheck.Checked := True;
	CompGitCheck.OnClick := @ComponentChanged;

	CompNodeCheck := TNewCheckBox.Create(ComponentsPanel);
	CompNodeCheck.Parent := ComponentsPanel;
	CompNodeCheck.Caption := 'Install Node.js LTS (if missing)';
	CompNodeCheck.Left := 0;
	CompNodeCheck.Top := CompGitCheck.Top + CompGitCheck.Height + ScaleY(4);
	CompNodeCheck.Width := ComponentsPanel.Width;
	CompNodeCheck.Height := ScaleY(20);
	CompNodeCheck.Checked := True;
	CompNodeCheck.OnClick := @ComponentChanged;

	CompCudaCheck := TNewCheckBox.Create(ComponentsPanel);
	CompCudaCheck.Parent := ComponentsPanel;
	CompCudaCheck.Caption := 'Install NVIDIA CUDA Tools (Force)';
	CompCudaCheck.Left := 0;
	CompCudaCheck.Top := CompNodeCheck.Top + CompNodeCheck.Height + ScaleY(4);
	CompCudaCheck.Width := ComponentsPanel.Width;
	CompCudaCheck.Height := ScaleY(20);
	CompCudaCheck.Checked := False;
	CompCudaCheck.OnClick := @ComponentChanged;

	DiskSummaryLabel := TNewStaticText.Create(ConfigPage.Surface);
	DiskSummaryLabel.Parent := ConfigPage.Surface;
	DiskSummaryLabel.Left := 0;
	DiskSummaryLabel.Width := ConfigPage.SurfaceWidth;
	DiskSummaryLabel.AutoSize := True;
	DiskSummaryLabel.WordWrap := True;
	DiskSummaryLabel.Font.Style := [fsBold];

	DiskDetailLabel := TNewStaticText.Create(ConfigPage.Surface);
	DiskDetailLabel.Parent := ConfigPage.Surface;
	DiskDetailLabel.Left := 0;
	DiskDetailLabel.Width := ConfigPage.SurfaceWidth;
	DiskDetailLabel.AutoSize := True;
	DiskDetailLabel.WordWrap := True;

	LogLabel := TNewStaticText.Create(ConfigPage.Surface);
	LogLabel.Parent := ConfigPage.Surface;
	LogLabel.Caption := 'Preflight status & log:';
	LogLabel.Font.Style := [fsBold];
	LogLabel.Left := 0;
	LogLabel.Width := ConfigPage.SurfaceWidth;

	LogViewer := TRichEditViewer.Create(ConfigPage.Surface);
	LogViewer.Parent := ConfigPage.Surface;
	LogViewer.Left := 0;
	LogViewer.Width := ConfigPage.SurfaceWidth;
	LogViewer.BorderStyle := bsSingle;
	LogViewer.ReadOnly := True;
	LogViewer.ScrollBars := ssVertical;
	LogViewer.WordWrap := False;
	LogViewer.Lines.Text := 'Waiting for disk usage preflight...';

	{ --- Install Page Controls --- }
	InstallDetailsButton := TNewButton.Create(WizardForm);
	InstallDetailsButton.Parent := WizardForm.InstallingPage;
	InstallDetailsButton.Caption := 'Show Details v';
	InstallDetailsButton.Left := 0;
	InstallDetailsButton.Top := WizardForm.ProgressGauge.Top + WizardForm.ProgressGauge.Height + ScaleY(10);
	InstallDetailsButton.Width := ScaleX(120);
	InstallDetailsButton.Height := ScaleY(30);
	InstallDetailsButton.OnClick := @InstallDetailsButtonClick;
	
	InstallLogViewer := TRichEditViewer.Create(WizardForm);
	InstallLogViewer.Parent := WizardForm.InstallingPage;
	InstallLogViewer.Left := 0;
	InstallLogViewer.Top := InstallDetailsButton.Top + InstallDetailsButton.Height + ScaleY(10);
	InstallLogViewer.Width := WizardForm.StatusLabel.Width;
	InstallLogViewer.Height := ScaleY(150);
	InstallLogViewer.ScrollBars := ssVertical;
	InstallLogViewer.BorderStyle := bsSingle;
	InstallLogViewer.ReadOnly := True;
	InstallLogViewer.Visible := False;
	InstallDetailsVisible := False;

	UpdateConfigPageLayout;
	StartPreflight(False);
	UpdateNextButtonState;
end;
'@

$issContent += "`r`n" + $codeBlock

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
	Write-Info "Cleaning build workspace (waiting for handles to release...)"
	Start-Sleep -Seconds 2
	Remove-PathRobust -Path $buildRoot
}

Write-Info "Done"