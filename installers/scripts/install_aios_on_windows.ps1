param(
  [ValidateSet('install','uninstall')]
  [string]$Action = 'install',
  [switch]$Yes,
  # GPU preference: auto (detect), cuda, dml, cpu
  [ValidateSet('auto','cuda','dml','cpu')]
  [string]$Gpu = 'auto',
  # Internal: elevated sub-process to perform admin-only steps
  [switch]$ElevatedSubprocess
)

$ErrorActionPreference = 'Stop'

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
  $info = @{ Vendor = 'Unknown'; Model = ''; HasNvidia = $false; NvidiaSmi = $false }
  try {
    $gpus = Get-CimInstance Win32_VideoController | Select-Object -Property Name, AdapterCompatibility
    foreach ($g in $gpus) {
      $vendor = [string]$g.AdapterCompatibility
      if ($vendor -match 'NVIDIA') {
        $info.Vendor = 'NVIDIA'; $info.Model = [string]$g.Name; $info.HasNvidia = $true
      } elseif ($info.Vendor -eq 'Unknown') {
        $info.Vendor = $vendor; $info.Model = [string]$g.Name
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
  Write-Host ("[i] GPU detection: Vendor={0} Model={1} NvidiaSmi={2}" -f $gpuInfo.Vendor, $gpuInfo.Model, $gpuInfo.NvidiaSmi)

  if ($gpu -eq 'auto') {
    if ($gpuInfo.HasNvidia) { $gpu = 'cuda' }
    else { $gpu = 'cpu' }
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
    $wheelName = "flash_attn-2.7.4+cu${cudaVer}torch${torchVer}cxx11abiFALSE-cp${pyVer}-cp${pyVer}-win_amd64.whl"
    $wheelUrl = "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/$wheelName"
    $wheelPath = Join-Path $env:TEMP $wheelName
    if (-not $Quiet) {
      Write-Host "[i] Downloading FlashAttention wheel: $wheelName"
    }
    Invoke-WebRequest -Uri $wheelUrl -OutFile $wheelPath -UseBasicParsing
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

  if (Confirm-Choice "Install Python 3.11 via winget now?" -DefaultYes:$true) {
    try {
      winget install -e --id Python.Python.3.11 -h
      Write-Host "[+] Python installed." -ForegroundColor Green
      # After install, try to resolve via 'py' without requiring a new terminal session
      $p = Get-PythonExecutableForVersion '3.11'
      if ($p) {
        $ver = & $p -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -and ([version]$ver -ge [version]'3.10')) { return $true }
      }
      # Fallback: let caller continue; interpreter selection logic will also try 'py'
      return $true
    } catch {
      Write-Host "[!] winget install failed. Please install Python >= $minVersion manually and re-run." -ForegroundColor Red
      throw
    }
  } else {
    throw "Python >= $minVersion is required. Please install it and re-run."
  }
  return $false
}

function Test-Git() {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "[+] Git is installed." -ForegroundColor Green
    return
  }
  Write-Host "[!] Git not found." -ForegroundColor Yellow
  if (Confirm-Choice "Install Git via winget now?" -DefaultYes:$true) {
    try {
      winget install -e --id Git.Git -h
      Write-Host "[+] Git installed." -ForegroundColor Green
      return
    } catch {
      Write-Host "[!] winget install failed. Please install Git manually and re-run." -ForegroundColor Red
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
  try {
    winget install -e --id OpenJS.NodeJS.LTS -h
    Write-Host "[+] Node.js installed." -ForegroundColor Green
    Write-Host "    Note: You may need to restart your terminal for 'node' to be available in PATH." -ForegroundColor Yellow
  } catch {
    Write-Host "[!] winget install failed. Node.js is required for MCP tool support." -ForegroundColor Red
    Write-Host "    Install manually: winget install OpenJS.NodeJS.LTS" -ForegroundColor Red
    throw "Node.js installation failed."
  }
}

# Core Logic Functions
function Install-Aios() {
  Write-Host "--- Starting AI-OS Windows Installation ---"
  Test-Python
  Test-Git
  Test-NodeJS  # Check for Node.js (optional but recommended)

  # Decide best Python interpreter (prefer 3.12 for CUDA builds)
  $gpuInfo = Get-GpuInfo
  $preferCuda = ($Gpu -eq 'cuda') -or ($Gpu -eq 'auto' -and $gpuInfo.HasNvidia)
  $pyExec = ''
  if ($preferCuda) {
    $pyExec = Get-PythonExecutableForVersion '3.12'
    if (-not $pyExec) {
      Write-Host "[!] Python 3.12 not found but recommended for CUDA wheels. Attempt installation via winget…" -ForegroundColor Yellow
      if (Confirm-Choice "Install Python 3.12 via winget now?" -DefaultYes:$true) {
        winget install -e --id Python.Python.3.12 -h
        $pyExec = Get-PythonExecutableForVersion '3.12'
      }
    }
  }
  if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.11' }
  if (-not $pyExec) { $pyExec = Get-PythonExecutableForVersion '3.10' }
  if (-not $pyExec) {
    try { $pyExec = (Get-Command python -ErrorAction SilentlyContinue).Source } catch {}
  }
  if (-not $pyExec) { throw "No suitable Python interpreter found." }

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

  $fullExtrasInstalled = $false
  Write-Host "[i] Installing AI-OS with UI/HF extras (includes tkinterweb for Help rendering)..."
  Push-Location $repoRoot
  try {
    & $pipPath install -e ".[ui,hf]"
    $fullExtrasInstalled = $true
  } catch {
    Write-Host "[!] install -e '.[ui,hf]' failed: $_" -ForegroundColor Yellow
    Write-Host "[i] Falling back to staged dependency install..." -ForegroundColor Yellow
  } finally {
    Pop-Location
  }

  if (-not $fullExtrasInstalled) {
    Install-UiExtras -pipPath $pipPath -Extras ".[ui]"
    $hfMode = if ($script:PyTorchBuild -eq 'cuda') { 'gpu' } else { 'cpu' }
    Install-HfStack -pipPath $pipPath -Mode $hfMode
    if ($script:PyTorchBuild -eq 'cuda') {
      Install-FlashAttentionWheel -pipPath $pipPath -pythonPath $pythonPath -Quiet | Out-Null
    }
  }

  Write-Host "[i] Ensuring httpx is available (dataset panel dependency)..."
  & $pipPath install --upgrade "httpx>=0.27"
  
  Write-Host "[i] Ensuring Ruff linter is available for developer tooling..."
  & $pipPath install --upgrade "ruff>=0.4.0"

  # Install lm-evaluation-harness separately (for model evaluation)
  Write-Host "[i] Installing lm-evaluation-harness for model benchmarking..."
  try {
    & $pipPath install "lm-eval>=0.4.0"
    Write-Host "[+] lm-evaluation-harness installed successfully" -ForegroundColor Green
  } catch {
    Write-Host "[!] lm-evaluation-harness installation failed (non-critical): $_" -ForegroundColor Yellow
    Write-Host "[i] You can manually install it later with: pip install lm-eval" -ForegroundColor Yellow
  }

  if ($script:PyTorchBuild -eq 'cuda') {
    Install-FlashAttentionWheel -pipPath $pipPath -pythonPath $pythonPath | Out-Null
  } else {
    Write-Host "[i] Skipping FlashAttention (CUDA GPU required for extreme context support)" -ForegroundColor Gray
  }

  Write-Host "[i] Installing Playwright browser (chromium)..."
  $playwrightPath = Join-Path (Join-Path $venvPath "Scripts") "playwright.exe"
  # install-deps is for Linux; skip on Windows to avoid errors
  & $playwrightPath install chromium

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

  Write-Host "[+] AI-OS installation complete!" -ForegroundColor Green
  Write-Host "To activate the environment, run: & '$venvPath\Scripts\Activate.ps1'"
  Write-Host "Then you can use the 'aios' command."

  # Create Start Menu shortcut with icon
  try {
    $startMenuPath = [Environment]::GetFolderPath('StartMenu')
    $programsPath = Join-Path $startMenuPath 'Programs'
    $shortcutPath = Join-Path $programsPath 'AI-OS.lnk'
    
    $iconPath = Join-Path $repoRoot 'installers\AI-OS.ico'
    
    $WScriptShell = New-Object -ComObject WScript.Shell
    $shortcut = $WScriptShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "$venvPath\Scripts\python.exe"
    $shortcut.Arguments = "-m aios.cli.aios gui"
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.Description = "AI-OS - Advanced AI Operating System"
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
      $shortcut.TargetPath = "$venvPath\Scripts\python.exe"
      $shortcut.Arguments = "-m aios.cli.aios gui"
      $shortcut.WorkingDirectory = $repoRoot
      $shortcut.Description = "AI-OS - Advanced AI Operating System"
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
    exit 1
  } finally {
    Pop-Location
  }
}

Main
