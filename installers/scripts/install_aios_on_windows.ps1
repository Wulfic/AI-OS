param(
  [ValidateSet('install','uninstall','preflight')]
  [string]$Action = 'install',
  [switch]$Yes,
  # GPU preference: auto (detect), cuda, rocm (AMD), xpu (Intel), dml (DirectML), cpu
  [ValidateSet('auto','cuda','rocm','xpu','dml','cpu')]
  [string]$Gpu = 'auto',
  # Force installation of CUDA tools even if no NVIDIA GPU is detected
  [switch]$InstallCudaTools,
  # Force installation of Intel Extension for PyTorch (XPU) for Intel GPUs
  [switch]$InstallIntelXpu,
  # Force installation of AMD DirectML for AMD GPUs
  [switch]$InstallAmdDml,
  # Skip automatic installation of Python
  [switch]$SkipPythonInstall,
  # Skip automatic installation of Git
  [switch]$SkipGitInstall,
  # Skip automatic installation of Node.js
  [switch]$SkipNodeInstall,
  # Internal: elevated sub-process to perform admin-only steps
  [switch]$ElevatedSubprocess,
  # Suppress Write-Host output for quiet automation flows
  [switch]$Quiet,
  # Optional file target for preflight key=value output
  [string]$PreflightOutput,
  # Optional log destination for installer orchestration
  [string]$InstallerLog,
  # Optional override for core payload size (bytes)
  [long]$PayloadBytes = 0,
  # Explicitly enable brain download (skips prompt)
  [switch]$DownloadBrain,
  # Explicitly disable brain download (skips prompt)
  [switch]$SkipBrain,
  # Preferred Python version (e.g., "3.10.11") - will install this version if Python needs to be installed
  [string]$PreferredPythonVersion = ''
)

# Initialize desktop log FIRST - before any errors can occur
$script:DesktopLogInitialized = $false
$script:DesktopLogPath = ''
try {
    $desktopPath = [Environment]::GetFolderPath('Desktop')
    $script:DesktopLogPath = Join-Path $desktopPath "AIOS_Installer.log"
    
    # Check if log already exists - if so, append; if not, create with header
    if (Test-Path $script:DesktopLogPath) {
        # Append session separator
        $separator = @"

========================================
New Session: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Action: $Action
Parameters: Gpu=$Gpu, SkipPythonInstall=$SkipPythonInstall, SkipGitInstall=$SkipGitInstall, SkipNodeInstall=$SkipNodeInstall
========================================

"@
        Add-Content -Path $script:DesktopLogPath -Value $separator -Encoding UTF8
    } else {
        # Create new log file with header
        $header = @"
AI-OS Installer Log
Created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
========================================

Session: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Action: $Action
Parameters: Gpu=$Gpu, SkipPythonInstall=$SkipPythonInstall, SkipGitInstall=$SkipGitInstall, SkipNodeInstall=$SkipNodeInstall
InstallerLog: $InstallerLog
PreflightOutput: $PreflightOutput
PayloadBytes: $PayloadBytes
PSScriptRoot: $PSScriptRoot
========================================

"@
        Set-Content -Path $script:DesktopLogPath -Value $header -Encoding UTF8 -Force
    }
    $script:DesktopLogInitialized = $true
} catch {
    # If we can't create desktop log, continue anyway
    # Try to create an error file on desktop to indicate the problem
    try {
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $errorPath = Join-Path $desktopPath "AIOS_LOG_CREATION_FAILED_$timestamp.txt"
        Set-Content -Path $errorPath -Value "Failed to create installer log: $_" -Force
    } catch {}
    $script:DesktopLogInitialized = $false
}

$ErrorActionPreference = 'Stop'

if ($InstallCudaTools) {
    if ($script:DesktopLogInitialized) { Add-Content -Path $script:DesktopLogPath -Value "[i] Forcing CUDA installation as requested." -Encoding UTF8 }
    Write-Host "[i] Forcing CUDA installation as requested." -ForegroundColor Cyan
    $Gpu = 'cuda'
}

if ($InstallIntelXpu) {
    if ($script:DesktopLogInitialized) { Add-Content -Path $script:DesktopLogPath -Value "[i] Forcing Intel XPU installation as requested." -Encoding UTF8 }
    Write-Host "[i] Forcing Intel XPU (Extension for PyTorch) installation as requested." -ForegroundColor Cyan
    $Gpu = 'xpu'
}

if ($InstallAmdDml) {
    if ($script:DesktopLogInitialized) { Add-Content -Path $script:DesktopLogPath -Value "[i] Forcing AMD DirectML installation as requested." -Encoding UTF8 }
    Write-Host "[i] Forcing AMD DirectML installation as requested." -ForegroundColor Cyan
    $Gpu = 'dml'
}

function Resolve-RepoRoot {
  param([string]$StartDirectory)

  try {
    $current = (Resolve-Path $StartDirectory).Path
  } catch {
    return (Split-Path -Parent $StartDirectory)
  }

  while ($true) {
    if (Test-Path (Join-Path $current 'pyproject.toml')) {
      return $current
    }
    $parent = Split-Path -Parent $current
    if (-not $parent -or $parent -eq $current) {
      break
    }
    $current = $parent
  }

  $fallback = Split-Path -Parent $StartDirectory
  if (-not $fallback) {
    $fallback = $StartDirectory
  }
  return $fallback
}

$repoRoot = Resolve-RepoRoot -StartDirectory $PSScriptRoot
$script:PyTorchBuild = 'cpu'
$script:InstallerLogInitialized = $false
# Desktop log variables already initialized at the top of the script

function Initialize-InstallerLog {
  # Desktop log was already initialized at script start
  # This function only handles the installer log parameter if provided
  
  if ([string]::IsNullOrWhiteSpace($InstallerLog)) { return }
  try {
    $dir = Split-Path -Parent $InstallerLog
    if ($dir -and -not (Test-Path -LiteralPath $dir)) {
      New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    New-Item -ItemType File -Path $InstallerLog -Force -Value '' | Out-Null
    $script:InstallerLogInitialized = $true
  } catch {
    # Ignore log initialization failures; script can continue without log
  }
}

function Write-InstallerLog {
  param([string]$Message)

  $timestamp = (Get-Date -Format 'HH:mm:ss')
  $line = "[{0}] {1}" -f $timestamp, $Message
  
  # Write to desktop log
  if ($script:DesktopLogInitialized) {
    try {
      Add-Content -Path $script:DesktopLogPath -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue
    } catch {
      # Ignore desktop log write failures
    }
  }

  # Write to installer log parameter if provided
  if ([string]::IsNullOrWhiteSpace($InstallerLog)) { return }
  if (-not $script:InstallerLogInitialized) {
    Initialize-InstallerLog
  }
  if (-not $script:InstallerLogInitialized) { return }

  # Retry logic to handle file contention with Inno Setup reader
  for ($i = 0; $i -lt 10; $i++) {
    try {
      Add-Content -Path $InstallerLog -Value $line -Encoding UTF8 -ErrorAction Stop
      break
    } catch {
      Start-Sleep -Milliseconds 100
    }
  }
}

# Capture original Write-Host for internal use
if (-not (Test-Path function:\Write-Host-Original)) {
    if (Test-Path function:\Write-Host) {
        Copy-Item function:\Write-Host function:\Write-Host-Original
    } else {
        function Write-Host-Original { Microsoft.PowerShell.Utility\Write-Host @args }
    }
}

function Write-Host {
    param(
        [Parameter(ValueFromPipeline = $true)]
        [object]$Object,
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$Objects,
        [ConsoleColor]$ForegroundColor,
        [ConsoleColor]$BackgroundColor,
        [switch]$NoNewline
    )

    # If not quiet, write to console
    if (-not $Quiet) {
        Write-Host-Original @PSBoundParameters
    }

    # Always log to file if configured
    if ($Object) {
        Write-InstallerLog "$Object"
    }
}

Initialize-InstallerLog
Write-InstallerLog ("Helper invoked with action '{0}'" -f $Action)

# Helper Functions
function Test-IsAdmin() {
  try {
    $wid = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal $wid
    return $pr.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
  } catch { return $false }
}

function Get-PythonExecutableForVersion([string]$Version) {
  try {
    $out = & py -$Version -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
  } catch {}

  # Fallback: Check common locations for py.exe if not in PATH
  $candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe",
    "$env:SystemRoot\py.exe"
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) {
      try {
        $out = & $c -$Version -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
      } catch {}
    }
  }

  return ''
}

function Confirm-Choice($Message, [switch]$DefaultYes) {
  if ($Yes) { return $true }
  if ($DefaultYes) { $default = 'Y/n' } else { $default = 'y/N' }
  Write-Host "$Message [$default]: " -NoNewline
  $resp = Read-Host
  if ([string]::IsNullOrWhiteSpace($resp)) { return [bool]$DefaultYes }
  return ($resp.ToLower() -match '^(y|yes)$')
}

# GPU / PyTorch helpers
function Get-GpuInfo() {
  $info = @{ 
    Vendor = 'Unknown'
    Model = ''
    HasNvidia = $false
    HasAmd = $false
    HasIntel = $false
    NvidiaSmi = $false 
  }
  
  try {
    $gpus = Get-CimInstance Win32_VideoController | Select-Object -Property Name, AdapterCompatibility
    foreach ($g in $gpus) {
      $vendor = [string]$g.AdapterCompatibility
      $model = [string]$g.Name
      
      if ($vendor -match 'NVIDIA' -or $model -match 'NVIDIA|GeForce|RTX|GTX') {
        $info.HasNvidia = $true
        if ($info.Vendor -eq 'Unknown') { $info.Vendor = 'NVIDIA'; $info.Model = $model }
      }
      elseif ($vendor -match 'AMD' -or $model -match 'AMD|Radeon|RX') {
        $info.HasAmd = $true
        if ($info.Vendor -eq 'Unknown') { $info.Vendor = 'AMD'; $info.Model = $model }
      }
      elseif ($vendor -match 'Intel' -or $model -match 'Intel.*Arc|Intel.*Xe|Arc A') {
        $info.HasIntel = $true
        if ($info.Vendor -eq 'Unknown') { $info.Vendor = 'Intel'; $info.Model = $model }
      }
      elseif ($info.Vendor -eq 'Unknown') {
        $info.Vendor = $vendor; $info.Model = $model
      }
    }
  } catch {}
  try { if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $info.NvidiaSmi = $true } } catch {}
  return $info
}

function Install-PyTorch([string]$pipPath, [string]$pythonPath, [string]$GpuPref) {
  $script:PyTorchBuild = 'cpu'
  $gpu = $GpuPref
  $gpuInfo = Get-GpuInfo
  Write-Host ("[i] GPU detection: Vendor={0} Model={1}" -f $gpuInfo.Vendor, $gpuInfo.Model)
  Write-Host ("[i] GPUs detected: NVIDIA={0} AMD={1} Intel={2}" -f $gpuInfo.HasNvidia, $gpuInfo.HasAmd, $gpuInfo.HasIntel)

  # Warn if multiple GPU vendors detected
  $vendorCount = @($gpuInfo.HasNvidia, $gpuInfo.HasAmd, $gpuInfo.HasIntel) | Where-Object { $_ -eq $true } | Measure-Object | Select-Object -ExpandProperty Count
  if ($vendorCount -gt 1) {
    Write-Host "[!] Multiple GPU vendors detected. Auto-selecting based on priority (NVIDIA > AMD > Intel)." -ForegroundColor Yellow
    Write-Host "    To override, rerun installer with Advanced options or use -InstallCudaTools/-InstallIntelXpu/-InstallAmdDml." -ForegroundColor Yellow
  }

  if ($gpu -eq 'auto') {
    if ($gpuInfo.HasNvidia) { 
      $gpu = 'cuda' 
    }
    elseif ($gpuInfo.HasAmd) {
      # ROCm on Windows is experimental, use DirectML as safer option
      Write-Host "[i] AMD GPU detected. Using DirectML (stable) instead of ROCm (experimental on Windows)."
      $gpu = 'dml'
    }
    elseif ($gpuInfo.HasIntel) {
      $gpu = 'xpu'
    }
    else { 
      $gpu = 'cpu' 
    }
  }

  if ($gpu -eq 'cuda') {
    Write-Host "[i] Installing PyTorch (CUDA build, cu121)…"
    try {
      & $pipPath install --upgrade pip wheel setuptools | Out-Null
      $cudaIndexes = @('cu121','cu124')
      foreach ($idx in $cudaIndexes) {
        Write-Host "[i] Trying CUDA index: $idx"
        try {
          & $pipPath install --upgrade --force-reinstall --no-cache-dir torch --index-url ("https://download.pytorch.org/whl/{0}" -f $idx)
          $code = "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0, 'cuda_version': getattr(torch.version, 'cuda', None)}))"
          $res = & $pythonPath -c $code
          Write-Host "[i] Probe: $res"
          if ($res -match 'cuda_version":\s*"?1[\d.]+') { # crude check that a CUDA runtime is linked
            Write-Host "[+] PyTorch installed (CUDA via $idx)." -ForegroundColor Green
            $script:PyTorchBuild = 'cuda'
            return
          }
        } catch {
          Write-Host "[!] Failed on index $idx, trying next…" -ForegroundColor Yellow
        }
      }
      throw "CUDA wheels not found for current Python; will fallback to CPU."
    } catch {
      Write-Host "[!] CUDA build install failed. Falling back to CPU build…" -ForegroundColor Yellow
    }
  }

  if ($gpu -eq 'rocm') {
    Write-Host "[i] Installing PyTorch with ROCm support (AMD GPUs)…"
    Write-Host "[!] Note: ROCm on Windows is experimental" -ForegroundColor Yellow
    try {
      & $pipPath install --upgrade pip wheel setuptools | Out-Null
      # ROCm 6.0 for Windows (if available)
      & $pipPath install torch --index-url https://download.pytorch.org/whl/rocm6.0
      $code = "import torch; rocm_ok = hasattr(torch.version, 'hip') and torch.version.hip is not None; print({'torch': torch.__version__, 'rocm': rocm_ok, 'cuda_available': torch.cuda.is_available()})"
      $res = & $pythonPath -c $code
      Write-Host "[+] PyTorch installed with ROCm support. Probe: $res" -ForegroundColor Green
      $script:PyTorchBuild = 'rocm'
      return
    } catch {
      Write-Host "[!] ROCm install failed. Falling back to DirectML…" -ForegroundColor Yellow
      $gpu = 'dml'
    }
  }

  if ($gpu -eq 'dml') {
    Write-Host "[i] Installing PyTorch (CPU) + torch-directml…"
    try {
      & $pipPath install --upgrade pip wheel setuptools | Out-Null
      & $pipPath install torch --index-url https://download.pytorch.org/whl/cpu
      & $pipPath install torch-directml
  $code = "import importlib, torch; ok = importlib.util.find_spec('torch_directml') is not None; print({'torch': torch.__version__, 'dml_available': ok})"
  $res = & $pythonPath -c $code
      Write-Host "[+] PyTorch installed with DirectML. Probe: $res" -ForegroundColor Green
      $script:PyTorchBuild = 'dml'
      return
    } catch {
      Write-Host "[!] DirectML install failed. Falling back to CPU build…" -ForegroundColor Yellow
    }
  }

  if ($gpu -eq 'xpu') {
    Write-Host "[i] Installing PyTorch + Intel Extension for PyTorch (XPU)…"
    try {
      & $pipPath install --upgrade pip wheel setuptools | Out-Null
      # CPU PyTorch first
      & $pipPath install torch --index-url https://download.pytorch.org/whl/cpu
      # Then Intel Extension
      & $pipPath install intel-extension-for-pytorch
      
      $code = "import torch; xpu_ok = hasattr(torch, 'xpu') and torch.xpu.is_available(); print({'torch': torch.__version__, 'xpu_available': xpu_ok, 'xpu_device_count': torch.xpu.device_count() if xpu_ok else 0})"
      $res = & $pythonPath -c $code
      Write-Host "[+] PyTorch installed with Intel XPU support. Probe: $res" -ForegroundColor Green
      $script:PyTorchBuild = 'xpu'
      return
    } catch {
      Write-Host "[!] Intel XPU install failed. Falling back to DirectML…" -ForegroundColor Yellow
      # Try DirectML as fallback for Intel GPUs on Windows
      try {
        & $pipPath install torch-directml
        $code = "import importlib, torch; ok = importlib.util.find_spec('torch_directml') is not None; print({'torch': torch.__version__, 'dml_available': ok})"
        $res = & $pythonPath -c $code
        Write-Host "[+] Fallback: PyTorch installed with DirectML. Probe: $res" -ForegroundColor Green
        $script:PyTorchBuild = 'dml'
        return
      } catch {
        Write-Host "[!] DirectML fallback also failed. Using CPU build…" -ForegroundColor Yellow
      }
    }
  }

  Write-Host "[i] Installing PyTorch (CPU build)…"
  & $pipPath install --upgrade pip wheel setuptools | Out-Null
  & $pipPath install --upgrade --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  $code = "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda_available': False, 'cuda_version': getattr(torch.version, 'cuda', None)}))"
  $res = & $pythonPath -c $code
  Write-Host "[+] PyTorch installed (CPU). Probe: $res" -ForegroundColor Green
}

function Install-UiExtras([string]$pipPath, [string]$Extras) {
  Push-Location $repoRoot
  try {
    & $pipPath install -e $Extras
  } finally {
    Pop-Location
  }
}

function Install-HfStack([string]$pipPath, [string]$Mode) {
  $baseDeps = @(
    'transformers>=4.41.0',
    'accelerate>=0.31.0',
    'sentencepiece>=0.1.99',
    'protobuf>=3.20.0',
    'peft>=0.11.1',
    'datasets>=2.14.0',
    'huggingface_hub>=0.19.0',
    'hf-xet>=1.0.0',
    'Pillow>=10.0.0',
    'tqdm>=4.65.0',
    'safetensors>=0.3.1',
    'lm-eval>=0.4.0'
  )
  foreach ($pkg in $baseDeps) {
    & $pipPath install --upgrade $pkg
  }

  if ($Mode -eq 'gpu') {
    $gpuDeps = @('deepspeed>=0.14.0','mpi4py>=4.0.0','bitsandbytes>=0.43.0')
    foreach ($pkg in $gpuDeps) {
      try {
        & $pipPath install --upgrade $pkg
      } catch {
        Write-Host "[!] Failed to install $pkg (non-critical): $_" -ForegroundColor Yellow
      }
    }
  }
}

function Install-FlashAttentionWheel([string]$pipPath, [string]$pythonPath, [switch]$Quiet) {
  if ($script:PyTorchBuild -ne 'cuda') {
    if (-not $Quiet) {
      Write-Host "[i] Skipping FlashAttention (CUDA build required)." -ForegroundColor Gray
    }
    return $false
  }

  try {
    & $pythonPath -c "import importlib, sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
      if (-not $Quiet) {
        Write-Host "[i] FlashAttention already installed." -ForegroundColor Gray
      }
      return $true
    }
  } catch {}

  if (-not $Quiet) {
    Write-Host "[i] Installing FlashAttention-2 (precompiled wheel for Windows)..."
  }
  try {
    $pyVer = & $pythonPath -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
    $torchInfo = & $pythonPath -c "import torch; tv=torch.__version__.split('+')[0]; cv=torch.version.cuda; print(f'{tv}|{cv}')"
    $parts = $torchInfo.Split('|')
    $torchVer = $parts[0]
    $cudaVer = $parts[1] -replace '\.', ''
    
    # Try exact CUDA version first, then fall back to cu124 (most common available wheel)
    $cudaVersionsToTry = @($cudaVer, '124')
    $wheelPath = $null
    $wheelName = $null
    $downloadSuccess = $false
    
    foreach ($cudaVersion in $cudaVersionsToTry) {
      if ($downloadSuccess) { break }
      $wheelName = "flash_attn-2.7.4+cu${cudaVersion}torch${torchVer}cxx11abiFALSE-cp${pyVer}-cp${pyVer}-win_amd64.whl"
      $wheelUrl = "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/$wheelName"
      $wheelPath = Join-Path $env:TEMP $wheelName
      if (-not $Quiet) {
        Write-Host "[i] Trying FlashAttention wheel: $wheelName"
      }
      try {
        $response = Invoke-WebRequest -Uri $wheelUrl -OutFile $wheelPath -UseBasicParsing -PassThru -ErrorAction Stop
        if (Test-Path -LiteralPath $wheelPath) {
          $downloadSuccess = $true
          if (-not $Quiet -and $cudaVersion -ne $cudaVer) {
            Write-Host "[i] Using cu${cudaVersion} wheel (cu${cudaVer} not available)" -ForegroundColor Yellow
          }
        }
      } catch {
        if (-not $Quiet) {
          Write-Host "[i] Wheel cu${cudaVersion} not available, trying next..." -ForegroundColor Gray
        }
        Remove-Item -Path $wheelPath -Force -ErrorAction SilentlyContinue
      }
    }
    
    if (-not $downloadSuccess) {
      throw "No compatible FlashAttention wheel found for torch${torchVer} (tried cu${cudaVer}, cu124)"
    }
    & $pipPath install $wheelPath | Out-Null
    $verify = & $pythonPath -c "from flash_attn import flash_attn_func; import flash_attn; print(f'FlashAttention {flash_attn.__version__} installed!')" 2>&1
    if (-not $Quiet) {
      Write-Host "[+] $verify" -ForegroundColor Green
      Write-Host "[+] Sliding window support enabled for extreme context lengths (10K-100K tokens)" -ForegroundColor Green
    }
    Remove-Item $wheelPath -Force -ErrorAction SilentlyContinue
    return $true
  } catch {
    if (-not $Quiet) {
      Write-Host "[!] FlashAttention installation failed (non-critical): $_" -ForegroundColor Yellow
      Write-Host "[i] AI-OS will work without FlashAttention, but extreme context lengths (>8K) may be limited." -ForegroundColor Yellow
    } else {
      Write-Host "[i] FlashAttention pre-install failed (non-critical): $_" -ForegroundColor Gray
    }
    return $false
  }
}

function Get-CorePayloadBytes([long]$ProvidedBytes) {
  if ($ProvidedBytes -gt 0) { return [int64]$ProvidedBytes }
  try {
    $pyprojectPath = Join-Path $repoRoot 'pyproject.toml'
    if (-not (Test-Path -LiteralPath $pyprojectPath)) { return [int64]$ProvidedBytes }
    $sum = Get-ChildItem -LiteralPath $repoRoot -Recurse -File -ErrorAction Stop |
      Measure-Object -Property Length -Sum
    if ($sum -and $sum.Sum) {
      return [int64]$sum.Sum
    }
  } catch {}
  return [int64]$ProvidedBytes
}

function Get-DependencyFootprint([string]$GpuPref) {
  $resolved = $GpuPref
  if ($resolved -eq 'auto') {
    $gpuInfo = Get-GpuInfo
    if ($gpuInfo.HasNvidia) {
      $resolved = 'cuda'
    } elseif ($gpuInfo.Vendor -match 'AMD|Intel') {
      $resolved = 'dml'
    } else {
      $resolved = 'cpu'
    }
  }

  $oneGiB = 1GB
  switch ($resolved) {
    'cuda' {
      $deps = 9 * $oneGiB # torch + CUDA wheels + nv libs + FlashAttention cache
      $buffer = 2 * $oneGiB
      $note = 'CUDA build (PyTorch + NVIDIA runtime + FlashAttention cache)'
    }
    'dml' {
      $deps = 6 * $oneGiB # CPU baseline + DirectML packages
      $buffer = 2 * $oneGiB
      $note = 'DirectML build (CPU baseline + torch-directml + shader cache)'
    }
    default {
      $deps = 4 * $oneGiB # CPU-only PyTorch + evaluation stack
      $buffer = 1 * $oneGiB
      $note = 'CPU build (PyTorch CPU + evaluation/runtime extras)'
      $resolved = 'cpu'
    }
  }

  return [ordered]@{
    Mode = $resolved
    DependencyBytes = [int64]$deps
    BufferBytes = [int64]$buffer
    Note = $note
  }
}

function Write-PreflightResult {
  param(
    [hashtable]$Data
  )

  $lines = @()
  foreach ($entry in $Data.GetEnumerator()) {
    $lines += ("{0}={1}" -f $entry.Key, $entry.Value)
  }

  if ([string]::IsNullOrWhiteSpace($PreflightOutput)) {
    $lines | ForEach-Object { Write-Output $_ }
    return
  }

  $dir = Split-Path -Parent $PreflightOutput
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }

  Set-Content -Path $PreflightOutput -Value $lines -Encoding UTF8
}

function Check-Prerequisites {
  $status = [ordered]@{
    python = "Missing"
    git = "Missing"
    node = "Missing"
  }

  # Python
  $py = Get-PythonExecutableForVersion '3.12'
  if (-not $py) { $py = Get-PythonExecutableForVersion '3.11' }
  if (-not $py) { $py = Get-PythonExecutableForVersion '3.10' }
  if (-not $py) { try { $py = (Get-Command python -ErrorAction SilentlyContinue).Source } catch {} }
  
  if ($py) { $status['python'] = "Found" }

  # Git
  if (Get-Command git -ErrorAction SilentlyContinue) { $status['git'] = "Found" }

  # Node
  if (Get-Command node -ErrorAction SilentlyContinue) { $status['node'] = "Found" }

  return $status
}

function Invoke-Preflight {
  Write-InstallerLog 'Starting disk usage preflight check.'
  try {
    $coreBytes = Get-CorePayloadBytes -ProvidedBytes $PayloadBytes
    Write-InstallerLog ("Core payload bytes resolved to {0}" -f $coreBytes)
    
    $prereqs = Check-Prerequisites
    Write-InstallerLog ("Prerequisites: Python={0}, Git={1}, Node={2}" -f $prereqs['python'], $prereqs['git'], $prereqs['node'])

    $footprint = Get-DependencyFootprint -GpuPref $Gpu
    if (-not $footprint) {
      Write-InstallerLog 'Unable to compute dependency footprint.'
      throw "Unable to determine dependency footprint for GPU preference '$Gpu'."
    }
    Write-InstallerLog ("Dependency footprint: mode={0} deps={1} buffer={2}" -f $footprint.Mode, $footprint.DependencyBytes, $footprint.BufferBytes)
    $totalBytes = $coreBytes + $footprint.DependencyBytes + $footprint.BufferBytes

    $result = [ordered]@{
      status = 'ok'
      mode = $footprint.Mode
      core_bytes = [int64]$coreBytes
      dependency_bytes = [int64]$footprint.DependencyBytes
      buffer_bytes = [int64]$footprint.BufferBytes
      total_bytes = [int64]$totalBytes
      note = $footprint.Note
      python_status = $prereqs['python']
      git_status = $prereqs['git']
      node_status = $prereqs['node']
      timestamp = (Get-Date -Format 'o')
    }

    Write-PreflightResult -Data $result
    Write-InstallerLog ("Total requirement computed: {0}" -f $totalBytes)
    Write-InstallerLog 'Disk usage preflight completed successfully.'
  } catch {
    Write-InstallerLog ("Disk usage preflight failed: {0}" -f $_)
    throw
  }
}

# Determines the best Python version to install based on:
# 1. PreferredPythonVersion parameter (if specified)
# 2. Default to 3.10.11 (minimum supported, widely compatible)
function Get-PreferredPythonVersion {
  # Use the command-line parameter if specified
  if (-not [string]::IsNullOrWhiteSpace($PreferredPythonVersion)) {
    Write-Host "[i] Using preferred Python version: $PreferredPythonVersion" -ForegroundColor Cyan
    return $PreferredPythonVersion
  }
  
  # Default to Python 3.10.11 - the minimum required version per pyproject.toml
  # This ensures maximum compatibility and matches the user's expected environment
  $defaultVersion = "3.10.11"
  Write-Host "[i] Using default Python version: $defaultVersion" -ForegroundColor Cyan
  return $defaultVersion
}

# Returns the major.minor version string for winget package ID (e.g., "3.10" from "3.10.11")
function Get-PythonMajorMinor {
  param([string]$FullVersion)
  
  if ([string]::IsNullOrWhiteSpace($FullVersion)) {
    return "3.10"
  }
  
  $parts = $FullVersion -split '\.'
  if ($parts.Count -ge 2) {
    return "$($parts[0]).$($parts[1])"
  }
  return "3.10"
}

# Direct Python download and installation (fallback when winget fails)
function Install-PythonDirect {
  param([string]$Version = "")
  
  # Determine the version to install
  if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = Get-PreferredPythonVersion
  }
  
  Write-Host "[i] Attempting direct Python $Version installation from python.org..." -ForegroundColor Cyan
  
  $pythonUrl = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"
  $tempInstaller = Join-Path $env:TEMP "python-$Version-installer.exe"
  $logFile = Join-Path $env:TEMP "python-$Version-install.log"
  
  # Parse major.minor version for path construction
  $versionParts = $Version -split '\.'
  $majorMinor = "$($versionParts[0])$($versionParts[1])"  # e.g., "310" for 3.10.11
  
  try {
    # Download Python installer
    Write-Host "[i] Downloading Python $Version installer..." -ForegroundColor Cyan
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $pythonUrl -OutFile $tempInstaller -UseBasicParsing -TimeoutSec 300
    
    if (-not (Test-Path $tempInstaller)) {
      throw "Download failed - installer not found"
    }
    
    $fileSize = (Get-Item $tempInstaller).Length / 1MB
    Write-Host "[+] Downloaded Python installer ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
    
    # Install Python quietly with verbose logging for detailed output
    Write-Host "[i] Installing Python $Version (quiet mode with logging)..." -ForegroundColor Cyan
    Write-Host "[i] Installation log: $logFile" -ForegroundColor Gray
    
    # Run the Python installer directly with -Wait to ensure we wait for completion
    # The Python installer (WiX bundle) requires: /quiet InstallAllUsers=1 PrependPath=1 etc.
    $installerArgs = @(
      '/quiet',
      '/log', "`"$logFile`"",
      'InstallAllUsers=1',
      'PrependPath=1',
      'Include_test=0',
      'Include_launcher=1',
      'Include_pip=1'
    )
    $installerArgsString = $installerArgs -join ' '
    Write-Host "[i] Running: $tempInstaller $installerArgsString" -ForegroundColor Gray
    
    # Start the installer process directly and wait for it
    $proc = Start-Process -FilePath $tempInstaller -ArgumentList $installerArgsString -PassThru -WindowStyle Hidden
    
    # Monitor log file and output progress (with 10-minute timeout)
    Write-Host "[i] Monitoring installation progress (this may take a few minutes)..." -ForegroundColor Cyan
    $lastPosition = 0
    $progressDots = 0
    $startTime = Get-Date
    $maxWaitMinutes = 10
    while (-not $proc.HasExited) {
      Start-Sleep -Milliseconds 500
      
      # Check for timeout
      $elapsed = (Get-Date) - $startTime
      if ($elapsed.TotalMinutes -gt $maxWaitMinutes) {
        Write-Host "`n[!] Installation timeout after $maxWaitMinutes minutes" -ForegroundColor Red
        try { $proc.Kill() } catch {}
        throw "Python installation timed out after $maxWaitMinutes minutes"
      }
      
      # Read new content from log file
      if (Test-Path $logFile) {
        try {
          $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
          if ($logContent -and $logContent.Length -gt $lastPosition) {
            $newContent = $logContent.Substring($lastPosition)
            $lastPosition = $logContent.Length
            
            # Parse and display relevant log lines
            $lines = $newContent -split "`r?`n"
            foreach ($line in $lines) {
              $line = $line.Trim()
              if ($line -and $line.Length -gt 0) {
                # Filter for informative messages
                if ($line -match 'Action\s+\d+:\d+:\d+:\s+(.+)' -or
                    $line -match 'Installing\s+(.+)' -or
                    $line -match 'Extracting\s+(.+)' -or
                    $line -match 'Creating\s+(.+)' -or
                    $line -match 'Registering\s+(.+)' -or
                    $line -match 'Configuring\s+(.+)' -or
                    $line -match 'Copying\s+(.+)' -or
                    $line -match 'Property.*=.*' -or
                    $line -match 'INSTALL\.' -or
                    $line -match 'MSI \(s\)') {
                  Write-Host "    [Python] $line" -ForegroundColor Gray
                }
              }
            }
          }
        } catch {
          # Ignore read errors, log may be locked
        }
      }
      
      # Show progress indicator
      $progressDots++
      if ($progressDots % 4 -eq 0) {
        Write-Host "." -NoNewline -ForegroundColor Cyan
      }
    }
    Write-Host "" # New line after dots
    
    # Wait for process to fully complete
    $proc.WaitForExit()
    
    # Output final log content
    if (Test-Path $logFile) {
      try {
        $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
        if ($logContent -and $logContent.Length -gt $lastPosition) {
          $newContent = $logContent.Substring($lastPosition)
          $lines = $newContent -split "`r?`n"
          foreach ($line in $lines) {
            $line = $line.Trim()
            if ($line -and $line.Length -gt 0 -and ($line -match 'Action|Installing|Complete|Success|Error|Failed')) {
              Write-Host "    [Python] $line" -ForegroundColor Gray
            }
          }
        }
      } catch {}
    }
    
    if ($proc.ExitCode -ne 0) {
      # Output full log on failure for debugging
      Write-Host "[!] Python installer failed. Full log:" -ForegroundColor Red
      if (Test-Path $logFile) {
        $fullLog = Get-Content $logFile -ErrorAction SilentlyContinue
        foreach ($line in $fullLog | Select-Object -Last 50) {
          Write-Host "    $line" -ForegroundColor Yellow
        }
      }
      throw "Python installer exited with code $($proc.ExitCode)"
    }
    
    Write-Host "[+] Python $Version installed successfully!" -ForegroundColor Green
    
    # Copy log to desktop for easy access
    try {
      $desktopPath = [Environment]::GetFolderPath('Desktop')
      $desktopLogDest = Join-Path $desktopPath "python_install.log"
      if (Test-Path $logFile) {
        Copy-Item $logFile $desktopLogDest -Force -ErrorAction SilentlyContinue
        Write-Host "[i] Python installation log saved to: $desktopLogDest" -ForegroundColor Cyan
      }
    } catch {}
    
    # Refresh PATH
    try {
      $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } catch {}
    
    # Add common Python locations (dynamic based on version)
    $pythonPaths = @(
      "$env:ProgramFiles\Python$majorMinor",
      "$env:ProgramFiles\Python$majorMinor\Scripts",
      "$env:LOCALAPPDATA\Programs\Python\Python$majorMinor",
      "$env:LOCALAPPDATA\Programs\Python\Python$majorMinor\Scripts",
      "$env:LOCALAPPDATA\Programs\Python\Launcher"
    )
    foreach ($pp in $pythonPaths) {
      if ((Test-Path $pp) -and $pp -notin $env:Path.Split(';')) {
        $env:Path += ";$pp"
      }
    }
    
    return $true
  } catch {
    Write-Host "[!] Direct Python installation failed: $_" -ForegroundColor Red
    # Try to copy log even on failure for debugging
    try {
      $desktopPath = [Environment]::GetFolderPath('Desktop')
      $desktopLogDest = Join-Path $desktopPath "python_install_FAILED.log"
      if (Test-Path $logFile) {
        Copy-Item $logFile $desktopLogDest -Force -ErrorAction SilentlyContinue
        Write-Host "[i] Python installation log saved to: $desktopLogDest" -ForegroundColor Yellow
      }
    } catch {}
    return $false
  } finally {
    if (Test-Path $tempInstaller) {
      Remove-Item $tempInstaller -Force -ErrorAction SilentlyContinue
    }
  }
}

# Prerequisite Check Functions
function Test-Python() {
  $minVersion = [version]"3.10"
  try {
    $pythonVersionStr = (python --version) -replace "Python ", ""
    $pythonVersion = [version]$pythonVersionStr
    if ($pythonVersion -ge $minVersion) {
      Write-Host "[+] Python $pythonVersion is installed." -ForegroundColor Green
      return $true
    } else {
      Write-Host "[!] Python $pythonVersion is installed, but >= $minVersion is required." -ForegroundColor Yellow
    }
  } catch {
    Write-Host "[!] Python not found." -ForegroundColor Yellow
  }

  # Try the Python launcher to find a suitable interpreter without relying on PATH
  try {
    $candidates = @('3.12','3.11','3.10')
    foreach ($v in $candidates) {
      $p = Get-PythonExecutableForVersion $v
      if ($p) {
        $ver = & $p -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) {
          $vObj = [version]$ver
          if ($vObj -ge $minVersion) {
            Write-Host "[+] Python $ver is available via 'py' at: $p" -ForegroundColor Green
            return $true
          }
        }
      }
    }
  } catch {}

  # Determine the version to install
  $targetVersion = Get-PreferredPythonVersion
  $targetMajorMinor = Get-PythonMajorMinor -FullVersion $targetVersion

  if (Confirm-Choice "Install Python $targetMajorMinor via winget now?" -DefaultYes:$true) {
    if ($SkipPythonInstall) {
      Write-Host "[i] Skipping Python installation as requested." -ForegroundColor Yellow
      return $false
    }
    try {
      Write-Host "[i] Attempting to install Python..."
      if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "[!] Winget not found. Downloading and installing Winget..." -ForegroundColor Yellow
        
        # Define download URLs for Winget dependencies
        # VCLibs 14.0.33728.0 - required minimum version for Winget 1.24.25200
        # Using direct GitHub mirror for correct version, fallback to aka.ms
        $vclibsUrl = "https://github.com/M1k3G0/Win10_LTSC_VP9_Installer/raw/refs/heads/master/Microsoft.VCLibs.140.00.UWPDesktop_14.0.33728.0_x64__8wekyb3d8bbwe.Appx"
        $vclibsFallbackUrl = "https://aka.ms/Microsoft.VCLibs.x64.14.00.Desktop.appx"
        $uiXamlUrl = "https://github.com/microsoft/microsoft-ui-xaml/releases/download/v2.8.6/Microsoft.UI.Xaml.2.8.x64.appx"
        # Direct download URL for WindowsAppRuntime (latest stable)
        # Note: Use /latest/ instead of /stable/ - the /stable/ URL returns HTML error pages
        $appRuntimeUrl = "https://aka.ms/windowsappsdk/1.8/latest/windowsappruntimeinstall-x64.exe"
        $wingetUrl = "https://github.com/microsoft/winget-cli/releases/download/v1.9.25200/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle"
        
        # Temporary download locations
        $tempDir = $env:TEMP
        $tempVcLibs = Join-Path $tempDir "vclibs_x64.appx"
        $tempUiXaml = Join-Path $tempDir "ui_xaml_x64.appx"
        $tempAppRuntime = Join-Path $tempDir "windowsappruntime_x64.exe"
        $tempWinget = Join-Path $tempDir "winget.msixbundle"
        
        # Helper function to download with timeout and retry
        function Download-WithRetry {
            param(
                [string]$Url,
                [string]$OutFile,
                [string]$Description,
                [int]$TimeoutSec = 120,
                [int]$MaxRetries = 3
            )
            
            $attempt = 0
            while ($attempt -lt $MaxRetries) {
                $attempt++
                try {
                    Write-Host "[i] Downloading $Description (attempt $attempt/$MaxRetries)..." -ForegroundColor Cyan
                    
                    # Use Start-BitsTransfer for better reliability with large files
                    if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
                        Start-BitsTransfer -Source $Url -Destination $OutFile -ErrorAction Stop
                    } else {
                        # Fallback to Invoke-WebRequest with timeout
                        $ProgressPreference = 'SilentlyContinue'
                        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -TimeoutSec $TimeoutSec -MaximumRedirection 5 -ErrorAction Stop
                    }
                    
                    if (Test-Path $OutFile) {
                        $fileSize = (Get-Item $OutFile).Length / 1MB
                        Write-Host "[+] Downloaded $Description successfully ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
                        return $true
                    }
                } catch {
                    Write-Host "[!] Download attempt $attempt failed: $_" -ForegroundColor Yellow
                    if ($attempt -lt $MaxRetries) {
                        $waitSec = [math]::Pow(2, $attempt)
                        Write-Host "[i] Retrying in $waitSec seconds..." -ForegroundColor Cyan
                        Start-Sleep -Seconds $waitSec
                    }
                }
            }
            
            throw "Failed to download $Description after $MaxRetries attempts"
        }
        
        try {
            # Check if VCLibs is already installed with sufficient version
            $vclibsInstalled = Get-AppxPackage -Name "Microsoft.VCLibs.140.00*" -ErrorAction SilentlyContinue
            $minVcLibsVersion = [version]"14.0.33728.0"
            
            if (-not $vclibsInstalled -or ([version]$vclibsInstalled.Version -lt $minVcLibsVersion)) {
                if ($vclibsInstalled) {
                    Write-Host "[!] VCLibs version $($vclibsInstalled.Version) is too old (need >= 14.0.33728.0)" -ForegroundColor Yellow
                }
                
                # Try primary URL (GitHub mirror with correct version)
                $downloadSuccess = $false
                try {
                    Download-WithRetry -Url $vclibsUrl -OutFile $tempVcLibs -Description "VCLibs 14.0.33728.0"
                    $downloadSuccess = $true
                } catch {
                    Write-Host "[!] Primary VCLibs source failed, trying fallback: $_" -ForegroundColor Yellow
                    try {
                        Download-WithRetry -Url $vclibsFallbackUrl -OutFile $tempVcLibs -Description "VCLibs (fallback)"
                        $downloadSuccess = $true
                    } catch {
                        throw "Failed to download VCLibs from any source: $_"
                    }
                }
                
                if ($downloadSuccess) {
                    Write-Host "[i] Installing VCLibs..." -ForegroundColor Cyan
                    Add-AppxPackage -Path $tempVcLibs -ForceApplicationShutdown -ErrorAction Stop
                    
                    # Verify the installed version
                    $vclibsInstalled = Get-AppxPackage -Name "Microsoft.VCLibs.140.00*" -ErrorAction SilentlyContinue
                    if ($vclibsInstalled) {
                        Write-Host "[+] VCLibs installed successfully (version $($vclibsInstalled.Version))" -ForegroundColor Green
                        if ([version]$vclibsInstalled.Version -lt $minVcLibsVersion) {
                            Write-Host "[!] Warning: Installed VCLibs version is still too old, Winget may fail" -ForegroundColor Yellow
                        }
                    }
                }
            } else {
                Write-Host "[i] VCLibs already installed (version $($vclibsInstalled.Version)), skipping..." -ForegroundColor Cyan
            }
            
            # Download and install UI.Xaml
            Download-WithRetry -Url $uiXamlUrl -OutFile $tempUiXaml -Description "UI.Xaml"
            Write-Host "[i] Installing UI.Xaml..." -ForegroundColor Cyan
            Add-AppxPackage -Path $tempUiXaml -ForceApplicationShutdown -ErrorAction Stop
            
            # Download and install WindowsAppRuntime (optional, enhances Winget 1.27+)
            # Made non-fatal because core Winget functionality works without it
            try {
                Download-WithRetry -Url $appRuntimeUrl -OutFile $tempAppRuntime -Description "WindowsAppRuntime" -TimeoutSec 180
                
                # Validate that the downloaded file is actually an executable (not HTML error page)
                if (Test-Path $tempAppRuntime) {
                    $fileBytes = [System.IO.File]::ReadAllBytes($tempAppRuntime)
                    if ($fileBytes.Length -lt 2 -or $fileBytes[0] -ne 0x4D -or $fileBytes[1] -ne 0x5A) {
                        # Not a valid PE executable (doesn't start with MZ)
                        throw "Downloaded file is not a valid Windows executable (likely an error page from the server)"
                    }
                }
                
                Write-Host "[i] Installing WindowsAppRuntime..." -ForegroundColor Cyan
                $proc = Start-Process -FilePath $tempAppRuntime -ArgumentList '--quiet' -Wait -PassThru -NoNewWindow
                if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1638) {
                    # 1638 = already installed (newer version exists)
                    Write-Host "[!] Warning: WindowsAppRuntime installer exited with code $($proc.ExitCode), but continuing anyway" -ForegroundColor Yellow
                } else {
                    Write-Host "[+] WindowsAppRuntime installed successfully" -ForegroundColor Green
                }
            } catch {
                Write-Host "[!] Warning: WindowsAppRuntime installation failed: $_" -ForegroundColor Yellow
                Write-Host "[!] Continuing without WindowsAppRuntime (Winget core features should still work)" -ForegroundColor Yellow
            }
            
            # Download and install Winget
            Download-WithRetry -Url $wingetUrl -OutFile $tempWinget -Description "Winget" -TimeoutSec 180
            Write-Host "[i] Installing Winget..." -ForegroundColor Cyan
            Add-AppxPackage -Path $tempWinget -ForceApplicationShutdown -ErrorAction Stop
            Write-Host "[+] Winget installed successfully!" -ForegroundColor Green
            
        } catch {
            Write-Host "[!] Failed to download/install Winget: $_" -ForegroundColor Red
            throw "Winget installation failed: $_"
        } finally {
            # Clean up downloaded files
            @($tempVcLibs, $tempUiXaml, $tempAppRuntime, $tempWinget) | ForEach-Object {
                if (Test-Path $_) {
                    Remove-Item $_ -Force -ErrorAction SilentlyContinue
                }
            }
        }
        
      # Refresh env to find winget - more comprehensive PATH refresh
      Write-Host "[i] Refreshing environment to locate winget..." -ForegroundColor Cyan
      Start-Sleep -Seconds 3
      
      # Multiple potential locations for winget
      $wingetPaths = @(
        "$env:LOCALAPPDATA\Microsoft\WindowsApps",
        "$env:ProgramFiles\WindowsApps\Microsoft.DesktopAppInstaller_*_x64__8wekyb3d8bbwe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\Microsoft.DesktopAppInstaller_8wekyb3d8bbwe"
      )
      
      foreach ($wp in $wingetPaths) {
        if ($wp -match '\*') {
          $resolved = Get-Item $wp -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
          if ($resolved -and $resolved -notin $env:Path.Split(';')) {
            $env:Path += ";$resolved"
          }
        } elseif ((Test-Path $wp) -and $wp -notin $env:Path.Split(';')) {
          $env:Path += ";$wp"
        }
      }
      
      # Also refresh from registry
      try {
        $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$machinePath;$userPath;$env:Path"
      } catch {}
      
      # Try to find winget with retry
      $wingetFound = $false
      for ($retry = 0; $retry -lt 5; $retry++) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
          $wingetFound = $true
          break
        }
        Write-Host "[i] Waiting for winget to become available (attempt $($retry + 1)/5)..." -ForegroundColor Cyan
        Start-Sleep -Seconds 2
      }
      
      if (-not $wingetFound) {
          Write-Host "[!] Winget not found in PATH after installation. Will try direct Python download." -ForegroundColor Yellow
          throw "Winget installed but not found in PATH - falling back to direct download."
      }
      
      # Update winget sources after fresh installation with verbose output
      Write-Host "[i] Initializing winget package sources (this may take a moment)..." -ForegroundColor Cyan
      try {
        # First, accept agreements
        $null = winget list --accept-source-agreements 2>&1
        Start-Sleep -Seconds 2
        
        # Then update sources
        $sourceOutput = winget source update 2>&1
        Write-Host "[i] Source update output: $sourceOutput" -ForegroundColor Gray
        Start-Sleep -Seconds 3
      } catch {
        Write-Host "[!] Source update had issues: $_" -ForegroundColor Yellow
      }
    }
    
    Write-Host "[i] Installing Python via winget..." -ForegroundColor Cyan
    
    # Try up to 3 times with source reset on failure
    $attempts = 0
    $maxAttempts = 3
    $success = $false
    
    while ($attempts -lt $maxAttempts -and -not $success) {
      $attempts++
      if ($attempts -gt 1) {
        Write-Host "[i] Retrying Python installation (attempt $attempts/$maxAttempts)..." -ForegroundColor Yellow
        Write-Host "[i] Resetting winget sources..." -ForegroundColor Yellow
        try {
          winget source reset --force 2>&1 | Out-Null
          Start-Sleep -Seconds 3
          winget source update 2>&1 | Out-Null
          Start-Sleep -Seconds 2
        } catch {
          Write-Host "[!] Source reset failed: $_" -ForegroundColor Yellow
        }
      }
      
      $wingetPkgId = "Python.Python.$targetMajorMinor"
      Write-Host "[i] Running: winget install $wingetPkgId --source winget..." -ForegroundColor Cyan
      Write-Host "[i] Winget will download and install Python - showing progress below..." -ForegroundColor Yellow
      
      # Run winget with verbose output (removed -h/hidden flag for visibility)
      # Use --source winget to avoid ambiguity with msstore source
      $wingetOutput = winget install -e --id $wingetPkgId --source winget --accept-source-agreements --accept-package-agreements --verbose 2>&1
      $exitCode = $LASTEXITCODE
      
      # Log the output with categorization
      if ($wingetOutput) {
        foreach ($line in $wingetOutput) {
          $lineStr = "$line".Trim()
          if ($lineStr) {
            # Colorize based on content
            if ($lineStr -match 'error|fail|exception' -and $lineStr -notmatch 'errorhandling') {
              Write-Host "    [Winget] $lineStr" -ForegroundColor Red
            } elseif ($lineStr -match 'warning|warn') {
              Write-Host "    [Winget] $lineStr" -ForegroundColor Yellow
            } elseif ($lineStr -match 'success|complete|installed|found') {
              Write-Host "    [Winget] $lineStr" -ForegroundColor Green
            } elseif ($lineStr -match 'download|progress|%') {
              Write-Host "    [Winget] $lineStr" -ForegroundColor Cyan
            } else {
              Write-Host "    [Winget] $lineStr" -ForegroundColor Gray
            }
          }
        }
      }
      Write-Host "[i] Winget exit code: $exitCode" -ForegroundColor Gray
      
      if ($exitCode -eq 0 -or $exitCode -eq -1978335189) {
        # -1978335189 = already installed
        $success = $true
      } elseif ($attempts -lt $maxAttempts) {
        Write-Host "[!] Winget install failed (exit code $exitCode), retrying..." -ForegroundColor Yellow
        Start-Sleep -Seconds 2
      }
    }
    
    if (-not $success) {
      Write-Host "[!] Winget Python installation failed after $maxAttempts attempts. Trying direct download..." -ForegroundColor Yellow
      throw "winget failed - falling back to direct download"
    }
    Write-Host "[+] Python installed via winget." -ForegroundColor Green
      
      # Refresh environment variables to find the new installation
      try {
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
      } catch {}

      # After install, try to resolve via 'py' without requiring a new terminal session
      $p = Get-PythonExecutableForVersion $targetMajorMinor
      if ($p) {
        $ver = & $p -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -and ([version]$ver -ge [version]'3.10')) { return $true }
      }
      # Fallback: let caller continue; interpreter selection logic will also try 'py'
      return $true
    } catch {
      # Winget failed - try direct Python download as fallback
      Write-Host "[!] Winget-based installation failed: $_" -ForegroundColor Yellow
      Write-Host "[i] Attempting direct Python download from python.org..." -ForegroundColor Cyan
      
      if (Install-PythonDirect -Version $targetVersion) {
        # Verify the installation
        $p = Get-PythonExecutableForVersion $targetMajorMinor
        if ($p) {
          $ver = & $p -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
          if ($LASTEXITCODE -eq 0 -and $ver -and ([version]$ver -ge [version]'3.10')) {
            Write-Host "[+] Python $ver installed via direct download." -ForegroundColor Green
            return $true
          }
        }
        
        # Try checking via python command directly
        $versionNoDot = $targetMajorMinor -replace '\.', ''
        try {
          $pythonPath = "$env:ProgramFiles\Python$versionNoDot\python.exe"
          if (Test-Path $pythonPath) {
            $ver = & $pythonPath -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver) {
              Write-Host "[+] Python $ver installed at $pythonPath" -ForegroundColor Green
              return $true
            }
          }
        } catch {}
        
        # Installation seemed to succeed, return true
        return $true
      }
      
      Write-Host "[!] All Python installation methods failed." -ForegroundColor Red
      Write-Host "[!] Please install Python >= $minVersion manually from https://www.python.org/downloads/" -ForegroundColor Red
      throw "Python installation failed via all methods"
    }
  } else {
    throw "Python >= $minVersion is required. Please install it and re-run."
  }
  return $false
}

# Check and install Microsoft Visual C++ Redistributable (required for PyTorch DLLs)
function Test-VCRedist() {
  Write-Host "[i] Checking for Microsoft Visual C++ Redistributable..." -ForegroundColor Cyan
  
  # Check for VC++ Redistributable 2015-2022 (x64) - required for PyTorch
  # These are the registry keys where VC++ Redist is registered
  $vcRedistKeys = @(
    'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
    'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64'
  )
  
  $vcInstalled = $false
  $installedVersion = $null
  
  foreach ($key in $vcRedistKeys) {
    try {
      if (Test-Path $key) {
        $regValue = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
        if ($regValue -and $regValue.Installed -eq 1) {
          $vcInstalled = $true
          $installedVersion = $regValue.Version
          break
        }
      }
    } catch {}
  }
  
  # Alternative check: look for the actual DLLs
  if (-not $vcInstalled) {
    $vcruntime = "$env:SystemRoot\System32\vcruntime140.dll"
    $vcruntime_1 = "$env:SystemRoot\System32\vcruntime140_1.dll"
    if ((Test-Path $vcruntime) -and (Test-Path $vcruntime_1)) {
      $vcInstalled = $true
      $installedVersion = "Unknown (DLL present)"
    }
  }
  
  if ($vcInstalled) {
    Write-Host "[+] Microsoft Visual C++ Redistributable is installed (Version: $installedVersion)" -ForegroundColor Green
    return $true
  }
  
  Write-Host "[!] Microsoft Visual C++ Redistributable not found." -ForegroundColor Yellow
  Write-Host "[!] This is REQUIRED for PyTorch to work on Windows." -ForegroundColor Yellow
  
  # Always install VC++ Redistributable automatically (respects -Yes flag)
  # This is critical for PyTorch - do not skip this step
  $shouldInstall = $true
  if (-not $Yes) {
    $shouldInstall = Confirm-Choice "Install Microsoft Visual C++ Redistributable now?" -DefaultYes:$true
  }
  
  if ($shouldInstall) {
    $installSuccess = $false
    
    # Download URL for VC++ Redistributable 2015-2022 (x64)
    $vcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    $vcRedistPath = Join-Path $env:TEMP "vc_redist.x64.exe"
    
    Write-Host "[i] Downloading Visual C++ Redistributable 2015-2022 (x64)..." -ForegroundColor Cyan
    
    # Clean up any previous download
    if (Test-Path $vcRedistPath) {
      Remove-Item $vcRedistPath -Force -ErrorAction SilentlyContinue
    }
    
    # Method 1: Invoke-WebRequest
    $downloadSuccess = $false
    try {
      $ProgressPreference = 'SilentlyContinue'
      Invoke-WebRequest -Uri $vcRedistUrl -OutFile $vcRedistPath -UseBasicParsing -TimeoutSec 180
      if (Test-Path $vcRedistPath) {
        $fileSize = (Get-Item $vcRedistPath).Length
        if ($fileSize -gt 1000000) {  # At least 1MB
          Write-Host "[+] Downloaded VC++ Redistributable ($([math]::Round($fileSize / 1MB, 1)) MB)" -ForegroundColor Green
          $downloadSuccess = $true
        } else {
          throw "Downloaded file too small - likely incomplete"
        }
      }
    } catch {
      Write-Host "[!] Download method 1 failed: $_" -ForegroundColor Yellow
    }
    
    # Method 2: Start-BitsTransfer
    if (-not $downloadSuccess) {
      try {
        Write-Host "[i] Trying alternative download method (BITS)..." -ForegroundColor Cyan
        Start-BitsTransfer -Source $vcRedistUrl -Destination $vcRedistPath -ErrorAction Stop
        if (Test-Path $vcRedistPath) {
          $downloadSuccess = $true
        }
      } catch {
        Write-Host "[!] Download method 2 failed: $_" -ForegroundColor Yellow
      }
    }
    
    # Method 3: .NET WebClient
    if (-not $downloadSuccess) {
      try {
        Write-Host "[i] Trying fallback download method (.NET)..." -ForegroundColor Cyan
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($vcRedistUrl, $vcRedistPath)
        if (Test-Path $vcRedistPath) {
          $downloadSuccess = $true
        }
      } catch {
        Write-Host "[!] All download methods failed: $_" -ForegroundColor Red
      }
    }
    
    # Try to install if we have the file
    if ($downloadSuccess -and (Test-Path $vcRedistPath)) {
      try {
        Write-Host "[i] Installing Visual C++ Redistributable..." -ForegroundColor Cyan
        
        # Run installer silently - we're already running as admin from Inno Setup
        $proc = Start-Process -FilePath $vcRedistPath -ArgumentList '/install', '/quiet', '/norestart' -PassThru -Wait -ErrorAction Stop
        
        if ($proc.ExitCode -eq 0) {
          Write-Host "[+] Visual C++ Redistributable installed successfully!" -ForegroundColor Green
          $installSuccess = $true
        } elseif ($proc.ExitCode -eq 1638) {
          Write-Host "[+] A newer version of Visual C++ Redistributable is already installed." -ForegroundColor Green
          $installSuccess = $true
        } elseif ($proc.ExitCode -eq 3010) {
          Write-Host "[+] Visual C++ Redistributable installed. A reboot may be required." -ForegroundColor Yellow
          $installSuccess = $true
        } elseif ($proc.ExitCode -eq 5) {
          Write-Host "[!] Access denied - trying with elevation..." -ForegroundColor Yellow
          # Try with explicit elevation
          try {
            $proc = Start-Process -FilePath $vcRedistPath -ArgumentList '/install', '/quiet', '/norestart' -PassThru -Wait -Verb RunAs
            if ($proc.ExitCode -eq 0 -or $proc.ExitCode -eq 1638 -or $proc.ExitCode -eq 3010) {
              Write-Host "[+] Visual C++ Redistributable installed with elevation." -ForegroundColor Green
              $installSuccess = $true
            }
          } catch {
            Write-Host "[!] Failed to elevate VC++ installer: $_" -ForegroundColor Red
          }
        } else {
          Write-Host "[!] VC++ Redistributable installer exited with code $($proc.ExitCode)" -ForegroundColor Yellow
        }
        
        Remove-Item $vcRedistPath -Force -ErrorAction SilentlyContinue
      } catch {
        Write-Host "[!] Failed to run VC++ installer: $_" -ForegroundColor Red
        Remove-Item $vcRedistPath -Force -ErrorAction SilentlyContinue
      }
    }
    
    if (-not $installSuccess) {
      Write-Host "" -ForegroundColor Yellow
      Write-Host "===============================================================================" -ForegroundColor Red
      Write-Host " IMPORTANT: Visual C++ Redistributable installation may have failed!" -ForegroundColor Red
      Write-Host " PyTorch requires this component to run properly on Windows." -ForegroundColor Red
      Write-Host "" -ForegroundColor Red
      Write-Host " Please download and install manually from:" -ForegroundColor Yellow
      Write-Host "   https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Cyan
      Write-Host "" -ForegroundColor Yellow
      Write-Host " After installing, restart the AI-OS application." -ForegroundColor Yellow
      Write-Host "===============================================================================" -ForegroundColor Red
      Write-Host ""
    }
    
    return $installSuccess
  } else {
    Write-Host "[!] WARNING: PyTorch WILL NOT WORK without Visual C++ Redistributable!" -ForegroundColor Red
    Write-Host "[!] You MUST install it from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
    return $false
  }
}

function Test-Git() {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "[+] Git is installed." -ForegroundColor Green
    return
  }
  Write-Host "[!] Git not found." -ForegroundColor Yellow
  if (Confirm-Choice "Install Git via winget now?" -DefaultYes:$true) {
    if ($SkipGitInstall) {
      Write-Host "[i] Skipping Git installation as requested." -ForegroundColor Yellow
      return
    }
    try {
      if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "[!] winget not found. Cannot install Git automatically." -ForegroundColor Red
        throw "winget not found"
      }
      winget install -e --id Git.Git -h --accept-source-agreements --accept-package-agreements
      Write-Host "[+] Git installed." -ForegroundColor Green
      try {
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
      } catch {}
      return
    } catch {
      Write-Host "[!] Git installation failed. Please install Git manually and re-run." -ForegroundColor Red
      throw
    }
  } else {
    throw "Git is required. Install it and re-run."
  }
}

function Test-NodeJS() {
  # Check if Node.js is installed
  if (Get-Command node -ErrorAction SilentlyContinue) {
    try {
      $nodeVer = & node --version 2>$null
      Write-Host "[+] Node.js is installed: $nodeVer" -ForegroundColor Green
      return
    } catch {}
  }
  
  Write-Host "[!] Node.js not found. Installing Node.js LTS via winget..." -ForegroundColor Yellow
  if ($SkipNodeInstall) {
    Write-Host "[i] Skipping Node.js installation as requested." -ForegroundColor Yellow
    return
  }
  try {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "[!] winget not found. Cannot install Node.js automatically." -ForegroundColor Red
        throw "winget not found"
    }
    winget install -e --id OpenJS.NodeJS.LTS -h --accept-source-agreements --accept-package-agreements
    Write-Host "[+] Node.js installed." -ForegroundColor Green
    try {
      $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } catch {}
    Write-Host "    Note: You may need to restart your terminal for 'node' to be available in PATH." -ForegroundColor Yellow
  } catch {
    Write-Host "[!] Node.js installation failed. Node.js is required for MCP tool support." -ForegroundColor Red
    Write-Host "    Install manually: winget install OpenJS.NodeJS.LTS" -ForegroundColor Red
    throw "Node.js installation failed."
  }
}

function Install-LauncherWrapper {
  $wrapperSrc = Join-Path $repoRoot "installers\wrapper\AIOSLauncher.cs"
  $manifest = Join-Path $repoRoot "installers\wrapper\app.manifest"
  $exePath = Join-Path $repoRoot "AI-OS.exe"
  
  if (Test-Path $wrapperSrc) {
    Write-Host "[i] Compiling/Installing admin wrapper..."
    try {
      $csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
      if (Test-Path $csc) {
        & $csc /target:exe "/out:$exePath" "/win32manifest:$manifest" $wrapperSrc | Out-Null
        if (Test-Path $exePath) {
           Write-Host "[+] Wrapper compiled successfully." -ForegroundColor Green
           return $exePath
        }
      } else {
         Write-Host "[!] csc.exe not found at expected location." -ForegroundColor Yellow
      }
    } catch {
      Write-Host "[!] Failed to compile wrapper: $_" -ForegroundColor Yellow
    }
  }
  return $null
}

# Core Logic Functions
function Install-Aios() {
  Write-Host "--- Starting AI-OS Windows Installation ---"
  
  # Set PowerShell execution policy for current user to allow running scripts
  # This is needed for the user to run PowerShell scripts and profiles
  try {
    $currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
    if ($currentPolicy -eq 'Restricted' -or $currentPolicy -eq 'Undefined') {
      Write-Host "[i] Setting PowerShell execution policy to RemoteSigned for current user..." -ForegroundColor Cyan
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force -ErrorAction SilentlyContinue
      Write-Host "[+] PowerShell execution policy updated." -ForegroundColor Green
    }
  } catch {
    Write-Host "[!] Could not set PowerShell execution policy: $_" -ForegroundColor Yellow
    Write-Host "[!] You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
  }
  
  # CRITICAL: Install Visual C++ Redistributable FIRST - required for PyTorch DLLs
  Test-VCRedist
  
  Test-Python
  Test-Git
  Test-NodeJS  # Check for Node.js (optional but recommended)

  # Decide best Python interpreter
  # Prefer higher versions for better CUDA wheel support, but respect PreferredPythonVersion if set
  $gpuInfo = Get-GpuInfo
  $preferCuda = ($Gpu -eq 'cuda') -or ($Gpu -eq 'auto' -and $gpuInfo.HasNvidia)
  $pyExec = ''
  
  # Determine target version for installation if needed
  $targetVersion = Get-PreferredPythonVersion
  $targetMajorMinor = Get-PythonMajorMinor -FullVersion $targetVersion
  
  if ($preferCuda) {
    # For CUDA, try higher versions first as they have better wheel support
    $pyExec = Get-PythonExecutableForVersion '3.12'
    if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.11' }
    if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.10' }
    
    if (-not $pyExec) {
      Write-Host "[!] No suitable Python found. Will install Python $targetMajorMinor." -ForegroundColor Yellow
      $wingetPkgId = "Python.Python.$targetMajorMinor"
      if (Confirm-Choice "Install Python $targetMajorMinor via winget now?" -DefaultYes:$true) {
        winget install -e --id $wingetPkgId -h --accept-source-agreements --accept-package-agreements
        $pyExec = Get-PythonExecutableForVersion $targetMajorMinor
      }
    }
  }
  if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.12' }
  if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.11' }
  if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.10' }
  if (-not $pyExec) {
    try { $pyExec = (Get-Command python -ErrorAction SilentlyContinue).Source } catch {}
  }
  if (-not $pyExec) {
    if ($SkipPythonInstall) {
      Write-Host "[!] Python not found and installation skipped." -ForegroundColor Yellow
      Write-Host "[!] Skipping environment setup. You must configure AI-OS manually." -ForegroundColor Yellow
      
      # Create the shortcut pointing to the wrapper (or nothing?)
      # If we return here, we skip the rest of Install-Aios, including shortcut creation.
      # We should probably still create the shortcut but point it to a placeholder or just the wrapper.
      
      $wrapperExe = Install-LauncherWrapper
      if ($wrapperExe) {
          # Create shortcut to wrapper
          try {
            $startMenuPath = [Environment]::GetFolderPath('StartMenu')
            $programsPath = Join-Path $startMenuPath 'Programs'
            $shortcutPath = Join-Path $programsPath 'AI-OS.lnk'
            $iconPath = Join-Path $repoRoot 'installers\AI-OS.ico'
            $WScriptShell = New-Object -ComObject WScript.Shell
            $shortcut = $WScriptShell.CreateShortcut($shortcutPath)
            $shortcut.TargetPath = $wrapperExe
            $shortcut.Arguments = "gui"
            $shortcut.WorkingDirectory = $repoRoot
            $shortcut.Description = "AI-OS - Artificially Intelligent Operating System"
            if (Test-Path $iconPath) { $shortcut.IconLocation = $iconPath }
            $shortcut.Save()
          } catch {}
      }
      
      return
    }
    throw "No suitable Python interpreter found."
  }

  $venvPath = Join-Path $repoRoot ".venv"
  Write-Host "[i] Setting up Python virtual environment in '$venvPath' using '$pyExec'..."
  if (-not (Test-Path -Path "$venvPath\pyvenv.cfg")) {
    & $pyExec -m venv $venvPath
  }
  $pipPath = Join-Path (Join-Path $venvPath "Scripts") "pip.exe"
  $pythonPath = Join-Path (Join-Path $venvPath "Scripts") "python.exe"

  # Intelligent PyTorch installation (CUDA/DirectML/CPU) BEFORE other deps to ensure correct wheel
  Install-PyTorch -pipPath $pipPath -pythonPath $pythonPath -GpuPref $Gpu

  if ($script:PyTorchBuild -eq 'cuda') {
    Install-FlashAttentionWheel -pipPath $pipPath -pythonPath $pythonPath -Quiet | Out-Null
  }

  Write-Host "[i] Installing AI-OS with UI/HF extras (includes tkinterweb for Help rendering)..."
  Write-Host "[i] Repository root: $repoRoot"
  
  # Verify pyproject.toml exists before attempting install
  $pyprojectPath = Join-Path $repoRoot 'pyproject.toml'
  if (-not (Test-Path $pyprojectPath)) {
    Write-Host "[!] pyproject.toml not found at: $pyprojectPath" -ForegroundColor Red
    throw "Cannot find pyproject.toml - installation files may be incomplete"
  }
  Write-Host "[i] Found pyproject.toml at: $pyprojectPath"
  
  Push-Location $repoRoot
  try {
    Write-Host "[i] Running: pip install -e `".[ui,hf]`""
    
    # Use Start-Process to capture output more reliably on Windows
    $tempStdout = [System.IO.Path]::GetTempFileName()
    $tempStderr = [System.IO.Path]::GetTempFileName()
    
    $pipProcess = Start-Process -FilePath $pipPath -ArgumentList 'install', '-e', '.[ui,hf]' -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempStdout -RedirectStandardError $tempStderr
    $pipExitCode = $pipProcess.ExitCode
    
    $stdoutContent = if (Test-Path $tempStdout) { Get-Content $tempStdout -Raw -ErrorAction SilentlyContinue } else { "" }
    $stderrContent = if (Test-Path $tempStderr) { Get-Content $tempStderr -Raw -ErrorAction SilentlyContinue } else { "" }
    
    # Clean up temp files
    Remove-Item $tempStdout -Force -ErrorAction SilentlyContinue
    Remove-Item $tempStderr -Force -ErrorAction SilentlyContinue
    
    if ($pipExitCode -ne 0) {
      Write-Host "[!] pip install failed with exit code $pipExitCode" -ForegroundColor Red
      if ($stdoutContent) {
        Write-Host "[!] pip stdout:" -ForegroundColor Red
        $stdoutContent -split "`n" | Select-Object -Last 30 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
      }
      if ($stderrContent) {
        Write-Host "[!] pip stderr:" -ForegroundColor Red
        $stderrContent -split "`n" | Select-Object -Last 30 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
      }
      # Extract the actual error for the exception message
      $errorLines = @()
      if ($stderrContent) { $errorLines += ($stderrContent -split "`n" | Where-Object { $_ -match 'ERROR|error:|Could not|No matching' } | Select-Object -Last 3) }
      if ($stdoutContent) { $errorLines += ($stdoutContent -split "`n" | Where-Object { $_ -match 'ERROR|error:|Could not|No matching' } | Select-Object -Last 3) }
      $errorSummary = if ($errorLines.Count -gt 0) { $errorLines -join " | " } else { "pip exit code: $pipExitCode" }
      throw "Failed to install AI-OS dependencies: $errorSummary"
    }
    Write-Host "[+] AI-OS package installed successfully" -ForegroundColor Green
  } catch {
    Write-Host "[!] install -e '.[ui,hf]' failed: $($_.Exception.Message)" -ForegroundColor Red
    throw "Failed to install AI-OS dependencies."
  } finally {
    Pop-Location
  }
  
  # Verify the aios module is importable
  Write-Host "[i] Verifying aios module installation..."
  try {
    # Use a temp file for the verification script to avoid quote escaping issues
    $verifyScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'
    $verifyStdout = [System.IO.Path]::GetTempFileName()
    $verifyStderr = [System.IO.Path]::GetTempFileName()
    
    # Write a simple verification script
    @"
import sys
try:
    import aios
    version = getattr(aios, '__version__', 'installed')
    print(f'aios version: {version}')
    sys.exit(0)
except Exception as e:
    import traceback
    print(f'Import failed: {e}', file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"@ | Out-File -FilePath $verifyScript -Encoding utf8
    
    $verifyProcess = Start-Process -FilePath $pythonPath -ArgumentList $verifyScript -NoNewWindow -Wait -PassThru -RedirectStandardOutput $verifyStdout -RedirectStandardError $verifyStderr
    
    $verifyOutContent = if (Test-Path $verifyStdout) { Get-Content $verifyStdout -Raw } else { "" }
    $verifyErrContent = if (Test-Path $verifyStderr) { Get-Content $verifyStderr -Raw } else { "" }
    
    # Clean up temp files
    Remove-Item -Path $verifyScript -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $verifyStdout -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $verifyStderr -Force -ErrorAction SilentlyContinue
    
    if ($verifyProcess.ExitCode -ne 0) {
      Write-Host "[!] aios module verification failed (exit code $($verifyProcess.ExitCode)):" -ForegroundColor Red
      if ($verifyOutContent) {
        Write-Host "    Stdout: $verifyOutContent" -ForegroundColor Red
      }
      if ($verifyErrContent) {
        Write-Host "    Error details:" -ForegroundColor Red
        $verifyErrContent -split "`n" | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
      }
      throw "aios module not properly installed"
    }
    Write-Host "[+] $($verifyOutContent.Trim())" -ForegroundColor Green
  } catch {
    Write-Host "[!] Failed to verify aios installation: $_" -ForegroundColor Red
    throw "aios module installation verification failed"
  }

  Write-Host "[i] Ensuring httpx is available (dataset panel dependency)..."
  try {
    $httpxStdout = [System.IO.Path]::GetTempFileName()
    $httpxStderr = [System.IO.Path]::GetTempFileName()
    $httpxProc = Start-Process -FilePath $pipPath -ArgumentList 'install', '--upgrade', 'httpx>=0.27' -NoNewWindow -Wait -PassThru -RedirectStandardOutput $httpxStdout -RedirectStandardError $httpxStderr
    Remove-Item -Path $httpxStdout, $httpxStderr -Force -ErrorAction SilentlyContinue
    if ($httpxProc.ExitCode -eq 0) {
      Write-Host "[+] httpx installed successfully" -ForegroundColor Green
    } else {
      Write-Host "[!] httpx installation returned exit code $($httpxProc.ExitCode) (non-critical)" -ForegroundColor Yellow
    }
  } catch {
    Write-Host "[!] httpx installation failed (non-critical): $_" -ForegroundColor Yellow
  }
  
  Write-Host "[i] Ensuring Ruff linter is available for developer tooling..."
  try {
    $ruffStdout = [System.IO.Path]::GetTempFileName()
    $ruffStderr = [System.IO.Path]::GetTempFileName()
    $ruffProc = Start-Process -FilePath $pipPath -ArgumentList 'install', '--upgrade', 'ruff>=0.4.0' -NoNewWindow -Wait -PassThru -RedirectStandardOutput $ruffStdout -RedirectStandardError $ruffStderr
    Remove-Item -Path $ruffStdout, $ruffStderr -Force -ErrorAction SilentlyContinue
    if ($ruffProc.ExitCode -eq 0) {
      Write-Host "[+] Ruff installed successfully" -ForegroundColor Green
    } else {
      Write-Host "[!] Ruff installation returned exit code $($ruffProc.ExitCode) (non-critical)" -ForegroundColor Yellow
    }
  } catch {
    Write-Host "[!] Ruff installation failed (non-critical): $_" -ForegroundColor Yellow
  }

  if ($script:PyTorchBuild -eq 'cuda') {
    Install-FlashAttentionWheel -pipPath $pipPath -pythonPath $pythonPath | Out-Null
  } else {
    Write-Host "[i] Skipping FlashAttention (CUDA GPU required for extreme context support)" -ForegroundColor Gray
  }

  # Playwright browser installation (optional - for web scraping features)
  # Wrapped in outer try/catch to handle any edge cases with command resolution
  try {
    Write-Host "[i] Installing Playwright browser (chromium)..."
    $playwrightPath = Join-Path (Join-Path $venvPath "Scripts") "playwright.exe"
    if (Test-Path -LiteralPath $playwrightPath) {
      try {
        # Use Start-Process to avoid PowerShell command resolution issues
        $playwrightResult = Start-Process -FilePath $playwrightPath -ArgumentList "install", "chromium" -Wait -NoNewWindow -PassThru -ErrorAction Stop
        if ($playwrightResult.ExitCode -eq 0) {
          Write-Host "[+] Playwright chromium installed successfully" -ForegroundColor Green
        } else {
          Write-Host "[!] Playwright browser installation returned exit code $($playwrightResult.ExitCode) (non-critical)" -ForegroundColor Yellow
        }
      } catch {
        Write-Host "[!] Playwright browser installation failed (non-critical): $_" -ForegroundColor Yellow
      }
    } else {
      Write-Host "[i] Playwright CLI not found in venv, skipping browser setup (non-critical)" -ForegroundColor Gray
    }
  } catch {
    Write-Host "[i] Playwright browser setup skipped due to error (non-critical): $_" -ForegroundColor Gray
  }

  # Install Microsoft MPI for DeepSpeed distributed training
  Write-Host "[i] Checking Microsoft MPI for DeepSpeed support..."
  $mpiInstalled = Test-Path (Join-Path $env:ProgramFiles "Microsoft MPI\Bin\mpiexec.exe")
  if (-not $mpiInstalled) {
    Write-Host "[i] Microsoft MPI not found. Installing for DeepSpeed distributed training..." -ForegroundColor Yellow
    try {
      $msMpiUrl = "https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe"
      $msMpiPath = Join-Path $env:TEMP "msmpisetup.exe"
      
      Write-Host "[i] Downloading Microsoft MPI..."
      Invoke-WebRequest -Uri $msMpiUrl -OutFile $msMpiPath -UseBasicParsing
      
      Write-Host "[i] Installing Microsoft MPI (silent install)..."
      Start-Process $msMpiPath -ArgumentList "/unattend" -Wait -NoNewWindow
      
      Remove-Item $msMpiPath -Force -ErrorAction SilentlyContinue
      Write-Host "[+] Microsoft MPI installed successfully" -ForegroundColor Green
    } catch {
      Write-Host "[!] Microsoft MPI installation failed (non-critical): $_" -ForegroundColor Yellow
      Write-Host "[i] DeepSpeed multi-GPU will require manual MPI installation" -ForegroundColor Yellow
      Write-Host "[i] Download from: https://github.com/microsoft/Microsoft-MPI/releases" -ForegroundColor Yellow
    }
  } else {
    Write-Host "[+] Microsoft MPI already installed" -ForegroundColor Green
  }

  # DeepSpeed Windows Patch
  Write-Host "[i] Checking DeepSpeed installation..."
  $deepspeedCheck = & $pythonPath -c "try: import deepspeed; print('installed')`nexcept: print('not-installed')" 2>$null
  if ($deepspeedCheck -eq 'installed') {
    Write-Host "[i] Applying DeepSpeed Windows patch..."
    $builderPath = Join-Path $venvPath "Lib\site-packages\deepspeed\ops\op_builder\builder.py"
    
    if (Test-Path $builderPath) {
      try {
        $content = Get-Content $builderPath -Raw
        
        # Check if already patched
        if ($content -notmatch 'Check if CUDA_HOME actually exists') {
          # Apply patch
          $oldCode = @'
def installed_cuda_version(name=""):
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise MissingCUDAException("CUDA_HOME does not exist, unable to compile CUDA op(s)")
    # Ensure there is not a cuda version mismatch between torch and nvcc compiler
    output = subprocess.check_output([cuda_home + "/bin/nvcc", "-V"], universal_newlines=True)
'@
          
          $newCode = @'
def installed_cuda_version(name=""):
    import torch.utils.cpp_extension
    import os
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    # Check if CUDA_HOME actually exists (for pre-built wheels)
    if cuda_home is None or not os.path.exists(cuda_home):
        # For pre-built wheels, return the torch CUDA version
        import torch
        torch_cuda_version = torch.version.cuda.split('.')
        return int(torch_cuda_version[0]), int(torch_cuda_version[1])
    # Ensure there is not a cuda version mismatch between torch and nvcc compiler
    output = subprocess.check_output([cuda_home + "/bin/nvcc", "-V"], universal_newlines=True)
'@
          
          $content = $content -replace [regex]::Escape($oldCode), $newCode
          Set-Content -Path $builderPath -Value $content -Encoding UTF8
          
          # Clear Python cache
          $deepspeedPath = Join-Path $venvPath "Lib\site-packages\deepspeed"
          Get-ChildItem -Path $deepspeedPath -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
          
          Write-Host "[+] DeepSpeed Windows patch applied successfully!" -ForegroundColor Green
          Write-Host "[i] DeepSpeed will now work without CUDA Toolkit installation" -ForegroundColor Green
        } else {
          Write-Host "[i] DeepSpeed patch already applied" -ForegroundColor Gray
        }
      } catch {
        Write-Host "[!] DeepSpeed patch failed (non-critical): $_" -ForegroundColor Yellow
        Write-Host "[i] DeepSpeed may still work if CUDA Toolkit is installed" -ForegroundColor Yellow
      }
    }
  } else {
    Write-Host "[i] DeepSpeed not installed, skipping patch" -ForegroundColor Gray
  }

  # Pre-create .lm_eval_cache to avoid permission issues
  Write-Host "[i] Configuring evaluation cache..."
  try {
    $userCache = [Environment]::GetFolderPath('LocalApplicationData')
    $evalCache = Join-Path $userCache "AI-OS\cache\lm_eval"
    if (-not (Test-Path $evalCache)) {
      New-Item -ItemType Directory -Path $evalCache -Force | Out-Null
    }
    Write-Host "[+] Evaluation cache directory created." -ForegroundColor Green
  } catch {
    Write-Host "[!] Failed to create evaluation cache (non-critical): $_" -ForegroundColor Yellow
  }

  # Verification Step
  Write-Host "[i] Verifying installation integrity..."
  try {
    & $pythonPath -c "import lm_eval; import tkinterweb; print('Core dependencies verified.')"
    if ($script:PyTorchBuild -eq 'cuda') {
        & $pythonPath -c "import flash_attn; print('FlashAttention verified.')"
    }
    Write-Host "[+] All critical dependencies verified." -ForegroundColor Green
  } catch {
    Write-Host "[!] Verification failed: $_" -ForegroundColor Red
    throw "Installation verification failed. Please check logs."
  }

  Write-Host "[+] AI-OS installation complete!" -ForegroundColor Green
  Write-Host "To activate the environment, run: & '$venvPath\Scripts\Activate.ps1'"
  Write-Host "Then you can use the 'aios' command."
  
  # Notify about desktop log location
  if ($script:DesktopLogInitialized -and (Test-Path $script:DesktopLogPath)) {
    Write-Host "[i] Installation log saved to: $($script:DesktopLogPath)" -ForegroundColor Cyan
  }

  # Compile wrapper
  $wrapperExe = Install-LauncherWrapper

  # Create Start Menu shortcut with icon
  try {
    $startMenuPath = [Environment]::GetFolderPath('StartMenu')
    $programsPath = Join-Path $startMenuPath 'Programs'
    $shortcutPath = Join-Path $programsPath 'AI-OS.lnk'
    
    $iconPath = Join-Path $repoRoot 'installers\AI-OS.ico'
    
    $WScriptShell = New-Object -ComObject WScript.Shell
    $shortcut = $WScriptShell.CreateShortcut($shortcutPath)
    
    if ($wrapperExe -and (Test-Path $wrapperExe)) {
        $shortcut.TargetPath = $wrapperExe
        $shortcut.Arguments = "gui"
    } else {
        $shortcut.TargetPath = "$venvPath\Scripts\python.exe"
        $shortcut.Arguments = "-m aios.cli.aios gui"
    }

    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.Description = "AI-OS - Artificially Intelligent Operating System"
    if (Test-Path $iconPath) {
      $shortcut.IconLocation = $iconPath
    }
    $shortcut.Save()
    
    Write-Host "[+] Created Start Menu shortcut: AI-OS" -ForegroundColor Green
  } catch {
    Write-Host "[!] Failed to create Start Menu shortcut (non-critical): $_" -ForegroundColor Yellow
  }

  # Create Desktop shortcut (optional, with user confirmation)
  if (Confirm-Choice "Create Desktop shortcut for AI-OS GUI?" -DefaultYes:$true) {
    try {
      $desktopPath = [Environment]::GetFolderPath('Desktop')
      $desktopShortcut = Join-Path $desktopPath 'AI-OS.lnk'
      
      $WScriptShell = New-Object -ComObject WScript.Shell
      $shortcut = $WScriptShell.CreateShortcut($desktopShortcut)
      
      if ($wrapperExe -and (Test-Path $wrapperExe)) {
          $shortcut.TargetPath = $wrapperExe
          $shortcut.Arguments = "gui"
      } else {
          $shortcut.TargetPath = "$venvPath\Scripts\python.exe"
          $shortcut.Arguments = "-m aios.cli.aios gui"
      }

      $shortcut.WorkingDirectory = $repoRoot
      $shortcut.Description = "AI-OS - Artificially Intelligent Operating System"
      if (Test-Path $iconPath) {
        $shortcut.IconLocation = $iconPath
      }
      $shortcut.Save()
      
      Write-Host "[+] Created Desktop shortcut: AI-OS" -ForegroundColor Green
    } catch {
      Write-Host "[!] Failed to create Desktop shortcut (non-critical): $_" -ForegroundColor Yellow
    }
  }

  # Add a user profile function so 'aios' works regardless of terminal CWD
  try {
    $start = '# BEGIN AIOS-LOCAL'
    $end = '# END AIOS-LOCAL'
    # Use a single-quoted here-string to avoid interpolating $true and $AiosArgs; replace placeholder with venv path
    $fn = @'
function aios {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AiosArgs)
  try {
    & '<<VENV>>\Scripts\python.exe' -m aios.cli.aios @AiosArgs
  } catch {
    Write-Host "[!] AI-OS not initialized. Attempting to re-install dependencies…" -ForegroundColor Yellow
    try { & '<<VENV>>\Scripts\pip.exe' install -e '.[ui,hf]' } catch {}
    & '<<VENV>>\Scripts\python.exe' -m aios.cli.aios @AiosArgs
  }
}
'@
    $fn = $fn.Replace('<<VENV>>', $venvPath)
    function Write-ProfileBlock([string]$profilePath) {
      if ([string]::IsNullOrWhiteSpace($profilePath)) { return }
      $dir = Split-Path $profilePath -ErrorAction SilentlyContinue
      if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
      $content = if (Test-Path $profilePath) { Get-Content $profilePath -Raw } else { '' }
      if ($content -match [regex]::Escape($start)) {
        $content = [regex]::Replace($content, "(?s)" + [regex]::Escape($start) + ".*?" + [regex]::Escape($end), '')
      }
      $content += "`n$start`n$fn$end`n"
      Set-Content -Path $profilePath -Value $content -Encoding UTF8
    }
    Write-ProfileBlock -profilePath $PROFILE.CurrentUserAllHosts
    $myDocs = [Environment]::GetFolderPath('MyDocuments')
    $winPs = Join-Path $myDocs 'WindowsPowerShell\profile.ps1'
    $pwsh  = Join-Path $myDocs 'PowerShell\profile.ps1'
    if ($pwsh -ne $PROFILE.CurrentUserAllHosts) { Write-ProfileBlock -profilePath $pwsh }
    Write-ProfileBlock -profilePath $winPs
    Write-Host "[i] Added user profile function 'aios' for easy access." -ForegroundColor Green
  } catch {}

  # Optional .cmd shim in WindowsApps for location-agnostic 'aios' in non-profile shells
  try {
    $apps = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps'
    if (Test-Path $apps) {
      $shim = Join-Path $apps 'aios.cmd'
      # Use PowerShell-escaped quotes for the path to python.exe
      $content = "@echo off`r`n`"$venvPath\Scripts\python.exe`" -m aios.cli.aios %*"
      Set-Content -Path $shim -Value $content -Encoding ASCII
      Write-Host "[i] Installed shim: $shim" -ForegroundColor Green
    }
  } catch {}

  # Optional Brain Download
  $doBrain = $false
  if ($DownloadBrain) { $doBrain = $true }
  elseif ($SkipBrain) { $doBrain = $false }
  else { $doBrain = Confirm-Choice "Download pretrained brain (English-v1)?" -DefaultYes:$true }

  if ($doBrain) {
    Write-Host "[i] Downloading pretrained brain..."
    try {
      & $pythonPath -m aios.cli.aios brains fetch --preset English-v1
      if ($LASTEXITCODE -eq 0) {
        Write-Host "[+] Brain downloaded successfully." -ForegroundColor Green
      } else {
        Write-Host "[!] Brain download failed (exit code $LASTEXITCODE)." -ForegroundColor Yellow
        Write-Host "[i] You can try again later with: aios brains fetch" -ForegroundColor Yellow
      }
    } catch {
      Write-Host "[!] Brain download failed: $_" -ForegroundColor Yellow
      Write-Host "[i] You can try again later with: aios brains fetch" -ForegroundColor Yellow
    }
  }
}

function Uninstall-Aios() {
  Write-Host "--- Starting AI-OS Windows Uninstallation ---"
  $venvPath = Join-Path $repoRoot ".venv"
  if (Test-Path -Path $venvPath) {
    if (Confirm-Choice "Remove virtual environment at '$venvPath'?" -DefaultYes:$true) {
      Write-Host "[i] Removing virtual environment..."
      Remove-Item -Path $venvPath -Recurse -Force
      Write-Host "[+] Virtual environment removed." -ForegroundColor Green
    }
  } else {
    Write-Host "[i] No virtual environment found at '$venvPath'."
  }
  Write-Host "[+] AI-OS uninstallation complete." -ForegroundColor Green
}

# Main Script Body
function Main() {
  if ($Action -eq 'preflight') {
    try {
      Invoke-Preflight
    } catch {
      $errorMessage = $_.Exception.Message
      $errorDetails = $_ | Out-String
      
      if ($Quiet) {
        Write-Error $_
      } else {
        Write-Host "[!!!] Preflight failed:" -ForegroundColor Red
        Write-Host $_ -ForegroundColor Red
      }

      # Emergency log - write directly to known locations regardless of $InstallerLog
      $emergencyLog = @"
[Preflight Failure Log]
Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Error: $errorMessage
Details:
$errorDetails

Script Path: $PSScriptRoot
Repo Root: $repoRoot
InstallerLog Parameter: $InstallerLog
Desktop Log: $($script:DesktopLogPath)
PayloadBytes Parameter: $PayloadBytes
Gpu Parameter: $Gpu
"@

      # Always notify about desktop log if it was created
      if ($script:DesktopLogInitialized -and (Test-Path $script:DesktopLogPath)) {
          if (-not $Quiet) {
              Write-Host "[i] Full log saved to desktop: $($script:DesktopLogPath)" -ForegroundColor Cyan
          }
      }

      $logDestinations = @("C:\AI-OS_Root", "C:\Installer", "$env:TEMP")
      foreach ($dest in $logDestinations) {
          if (Test-Path $dest) {
              try {
                  $emergencyLogPath = Join-Path $dest "aios_preflight_failure.log"
                  Set-Content -Path $emergencyLogPath -Value $emergencyLog -Force -Encoding UTF8
                  
                  # Also copy the InstallerLog if it exists
                  if ($InstallerLog -and (Test-Path $InstallerLog)) {
                      Copy-Item -Path $InstallerLog -Destination (Join-Path $dest "aios_preflight_detail.log") -Force
                  }
                  # Copy desktop log if it exists
                  if ($script:DesktopLogInitialized -and (Test-Path $script:DesktopLogPath)) {
                      Copy-Item -Path $script:DesktopLogPath -Destination (Join-Path $dest "aios_desktop.log") -Force
                  }
              } catch {}
          }
      }

      exit 1
    }
    return
  }

  # Elevate if required for future admin tasks (placeholder for now)
  if (-not (Test-IsAdmin) -and -not $ElevatedSubprocess) {
      # This section can be enabled if admin rights are needed, e.g., for firewall rules
      # Write-Host "[!] This script may require administrator privileges." -ForegroundColor Yellow
      # if (Confirm-Choice "Run required parts as Administrator?" -DefaultYes:$true) {
      #   $psISE = if ($psISE) { "-psISE" } else { "" }
      #   $params = "-NoProfile -File `"$($MyInvocation.MyCommand.Path)`" -Action $Action -ElevatedSubprocess $psISE"
      #   Start-Process pwsh -Verb RunAs -ArgumentList $params
      #   exit
      # }
  }

  Push-Location $repoRoot
  try {
    if ($Action -eq 'install') {
      Install-Aios
    } elseif ($Action -eq 'uninstall') {
      Uninstall-Aios
    }
  } catch {
    Write-Host "[!!!] An error occurred:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    
    # Always notify about desktop log if it was created
    if ($script:DesktopLogInitialized -and (Test-Path $script:DesktopLogPath)) {
        Write-Host "[i] Full log saved to desktop: $($script:DesktopLogPath)" -ForegroundColor Cyan
    }

    exit 1
  } finally {
    Pop-Location
  }
}

# Wrap Main execution to catch ANY error and log to desktop
try {
    Main
} catch {
    $errorMsg = "FATAL ERROR: $($_.Exception.Message)"
    $errorDetails = $_ | Out-String
    
    # Always try to write to desktop log
    if ($script:DesktopLogInitialized -and $script:DesktopLogPath) {
        try {
            $criticalError = @"

========================================
CRITICAL ERROR CAUGHT
========================================
$errorMsg

Full Details:
$errorDetails

Stack Trace:
$($_.ScriptStackTrace)
"@
            Add-Content -Path $script:DesktopLogPath -Value $criticalError -Encoding UTF8 -Force
            Write-Host "[!!!] Fatal error occurred. Check desktop log: $script:DesktopLogPath" -ForegroundColor Red
        } catch {
            # Last resort - can't even write to desktop log
        }
    }
    
    # Re-throw to preserve exit code
    throw
}
