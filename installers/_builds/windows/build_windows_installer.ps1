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

$stagingDir = (Resolve-Path $stagingDir).Path
$repoRoot = (Resolve-Path $repoRoot).Path
$releaseDir = (Resolve-Path $releaseDir).Path

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
LicenseFile={#RepoRoot}\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
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