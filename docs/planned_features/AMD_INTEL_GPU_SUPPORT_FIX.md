# AMD and Intel GPU Support Enhancement

**Status:** 📋 Planned  
**Priority:** High  
**Complexity:** Medium (4-8 hours implementation)  
**Target Version:** v1.1.0  
**Created:** 2025-10-23  
**Owner:** @AI-OS-Team

## Overview

Enhance detection, installation, and optimization support for AMD and Intel GPUs. Currently, AMD GPUs work but are mislabeled as "CUDA", and Intel XPU support is incomplete. This plan addresses all gaps to provide first-class support for non-NVIDIA GPUs.

## Problem Statement

### Current Issues

1. **AMD ROCm GPUs** - Detected but mislabeled as generic "CUDA" devices
2. **Intel Arc/Xe GPUs** - Basic support exists but memory optimization broken
3. **Installation Scripts** - Only detect NVIDIA, auto-install wrong PyTorch variant
4. **GUI** - Doesn't distinguish GPU vendors or show XPU devices
5. **Memory Optimizer** - Only checks CUDA, reports 0 VRAM for XPU-only systems

### Impact

- AMD GPU users confused about which hardware is being used
- Intel GPU users can't use memory optimization features
- Users must manually install correct PyTorch variant
- Sub-optimal experience for non-NVIDIA hardware

## Goals

### Must Have ✅

- [x] Correctly identify and label AMD GPUs
- [x] Add Intel XPU memory detection
- [x] Update installation scripts to detect all GPU vendors
- [x] Add `intel-extension-for-pytorch` to dependencies
- [x] Update GUI to show vendor information
- [x] Update `aios torch-info` command to report all vendors

### Nice to Have 🎯

- [ ] Vendor-specific optimization hints
- [ ] Auto-install correct PyTorch variant based on detected hardware
- [ ] XPU distributed training support (oneCCL backend)
- [ ] Performance benchmarking across vendors

### Out of Scope ❌

- Mixed-vendor parallel training (see `MIXED_GPU_VENDOR_SUPPORT.md`)
- Vendor-specific kernel optimizations
- GPU virtualization support
- Cloud GPU support (AWS/Azure specific configurations)

## Implementation Plan

### Phase 1: Core Detection (High Priority)

**Estimated Time:** 2 hours

#### Task 1.1: Add GPU Vendor Identification Function
**Files:** `src/aios/cli/training/torch_info_cmd.py`

- [ ] Create `identify_gpu_vendor()` function
- [ ] Check GPU name for vendor keywords (AMD, Radeon, MI, Intel, Arc, Xe, NVIDIA)
- [ ] Check `torch.version.hip` for ROCm builds
- [ ] Return vendor enum: `'NVIDIA' | 'AMD' | 'Intel' | 'Unknown'`

**Implementation:**
```python
def identify_gpu_vendor(gpu_name: str, check_rocm: bool = False) -> str:
    """Identify GPU vendor from device name.
    
    Args:
        gpu_name: GPU device name from torch
        check_rocm: Also check torch.version.hip attribute
        
    Returns:
        'NVIDIA' | 'AMD' | 'Intel' | 'Unknown'
    """
    name_lower = gpu_name.lower()
    
    # NVIDIA detection
    if any(kw in name_lower for kw in ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla', 'a100', 'h100']):
        return 'NVIDIA'
    
    # AMD detection
    if any(kw in name_lower for kw in ['amd', 'radeon', 'rx ', 'vega', 'navi', 'mi50', 'mi100', 'mi200', 'mi300']):
        return 'AMD'
    
    # Intel detection
    if any(kw in name_lower for kw in ['intel', 'arc', 'xe', 'iris', 'uhd']):
        return 'Intel'
    
    # Check ROCm build as fallback
    if check_rocm:
        try:
            import torch
            if getattr(torch.version, 'hip', None):
                return 'AMD'
        except Exception:
            pass
    
    return 'Unknown'
```

**Tests:**
- [ ] Test with NVIDIA GPU names (GeForce RTX 4090, Tesla V100)
- [ ] Test with AMD GPU names (Radeon RX 7900 XTX, MI210)
- [ ] Test with Intel GPU names (Arc A770, Xe HPG)
- [ ] Test ROCm detection fallback

#### Task 1.2: Update torch-info Command
**Files:** `src/aios/cli/training/torch_info_cmd.py`

- [ ] Import vendor identification function
- [ ] Add `vendor` field to each GPU in `cuda_devices` list
- [ ] Update `rocm` field to use vendor detection
- [ ] Add `gpu_vendor_summary` field with counts per vendor

**Changes:**
```python
# In torch_info() function, for each CUDA device:
dev: dict = {"id": i, "name": name}
if total_mb:
    dev["total_mem_mb"] = int(total_mb)

# ADD:
dev["vendor"] = identify_gpu_vendor(name, check_rocm=rocm)

cuda_devices.append(dev)

# ADD after device enumeration:
vendor_counts = {}
for dev in cuda_devices:
    vendor = dev.get("vendor", "Unknown")
    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1

info["gpu_vendor_summary"] = vendor_counts
```

**Tests:**
- [ ] Run `aios torch-info` with NVIDIA GPU
- [ ] Run with AMD GPU (if available)
- [ ] Run with Intel GPU (if available)
- [ ] Verify JSON output includes vendor field

#### Task 1.3: Update GUI Device Detection
**Files:** `src/aios/gui/app/panel_setup.py`

- [ ] Import vendor identification function
- [ ] Add `vendor` field to detected devices
- [ ] Update status label to show vendor counts
- [ ] Differentiate AMD from NVIDIA in device list

**Changes:**
```python
# In _detect_devices_info() function:
cuda_devices.append({
    "id": i,
    "name": name,
    "total_mem_mb": total_mem_mb,
    "vendor": identify_gpu_vendor(name, check_rocm=bool(getattr(torch.version, 'hip', None)))
})

# Update return dict:
return {
    "cuda_available": cuda_available,
    "cuda_devices": cuda_devices,
    "nvidia_smi_devices": cuda_devices,  # Keep for compatibility
    "vendor_summary": calculate_vendor_summary(cuda_devices)
}
```

**Tests:**
- [ ] Launch GUI and verify vendor shown in Resources panel
- [ ] Verify AMD GPUs labeled correctly
- [ ] Verify status message distinguishes vendors

### Phase 2: Intel XPU Support (High Priority)

**Estimated Time:** 3 hours

#### Task 2.1: Add IPEX Dependency
**Files:** `pyproject.toml`

- [ ] Add `intel-extension-for-pytorch` to `[project.optional-dependencies.hf]`
- [ ] Version constraint: `>=2.0.0`
- [ ] Platform constraint: exclude ARM64

**Changes:**
```toml
[project.optional-dependencies]
hf = [
  # ... existing dependencies ...
  "intel-extension-for-pytorch>=2.0.0; platform_machine != 'ARM64'",
]
```

**Tests:**
- [ ] Fresh install in new venv
- [ ] Verify IPEX installs on x64 systems
- [ ] Verify skipped on ARM64 systems

#### Task 2.2: Add XPU Memory Detection
**Files:** `src/aios/core/hrm_models/training_optimizer.py`

- [ ] Update `detect_available_vram()` to check XPU
- [ ] Add XPU device enumeration
- [ ] Handle IPEX not installed gracefully

**Implementation:**
```python
def detect_available_vram(self) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Detect available VRAM across all GPUs (CUDA, XPU).
    
    Returns:
        (total_vram_gb, list of GPU info dicts)
    """
    gpus = []
    total_vram = 0.0
    
    # Check CUDA devices (NVIDIA + AMD ROCm)
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_id)
            total_gb = props.total_memory / (1024 ** 3)
            
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            gc.collect()
            
            allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
            available_gb = total_gb - reserved_gb
            
            gpus.append({
                "id": gpu_id,
                "backend": "cuda",
                "name": props.name,
                "total_gb": total_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "available_gb": available_gb,
            })
            
            total_vram += total_gb
    
    # Check Intel XPU devices
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for gpu_id in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(gpu_id)
                total_gb = props.total_memory / (1024 ** 3)
                
                # XPU memory info
                try:
                    free_mem, total_mem = torch.xpu.mem_get_info(gpu_id)
                    available_gb = free_mem / (1024 ** 3)
                    reserved_gb = (total_mem - free_mem) / (1024 ** 3)
                except Exception:
                    available_gb = total_gb * 0.9  # Estimate
                    reserved_gb = total_gb * 0.1
                
                gpus.append({
                    "id": len(gpus),  # Global ID
                    "backend": "xpu",
                    "name": props.name,
                    "total_gb": total_gb,
                    "allocated_gb": 0.0,  # Not available via public API
                    "reserved_gb": reserved_gb,
                    "available_gb": available_gb,
                })
                
                total_vram += total_gb
    except ImportError:
        # intel-extension-for-pytorch not installed
        pass
    except Exception as e:
        # Log but don't crash
        print(f"XPU detection warning: {e}")
    
    return total_vram, gpus
```

**Tests:**
- [ ] Test with CUDA-only system (existing behavior)
- [ ] Test with XPU-only system (new behavior)
- [ ] Test with both CUDA and XPU
- [ ] Test without IPEX installed

#### Task 2.3: Update GUI XPU Device Enumeration
**Files:** `src/aios/gui/app/panel_setup.py`

- [ ] Add XPU detection to `_detect_devices_info()`
- [ ] Return separate `xpu_devices` list
- [ ] Update `set_detected()` to handle XPU

**Changes:**
```python
def _detect_devices_info() -> dict:
    try:
        import torch
        
        # ... existing CUDA detection ...
        
        # Detect Intel XPU devices
        xpu_available = False
        xpu_devices = []
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                xpu_available = True
                device_count = torch.xpu.device_count()
                for i in range(device_count):
                    try:
                        props = torch.xpu.get_device_properties(i)
                        total_mem_mb = props.total_memory // (1024 * 1024)
                        xpu_devices.append({
                            "id": i,
                            "name": props.name,
                            "total_mem_mb": total_mem_mb,
                            "vendor": "Intel"
                        })
                    except Exception:
                        xpu_devices.append({
                            "id": i,
                            "name": f"Intel XPU {i}",
                            "total_mem_mb": 0,
                            "vendor": "Intel"
                        })
        except Exception:
            xpu_available = False
        
        return {
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
            "nvidia_smi_devices": cuda_devices,
            "xpu_available": xpu_available,
            "xpu_devices": xpu_devices,
            # ... existing fields ...
        }
    except Exception:
        return {
            "cuda_available": False,
            "cuda_devices": [],
            "nvidia_smi_devices": [],
            "xpu_available": False,
            "xpu_devices": []
        }
```

**Tests:**
- [ ] GUI shows XPU devices when available
- [ ] XPU devices selectable in Resources panel
- [ ] Training can target XPU devices

#### Task 2.4: Update device.py for XPU
**Files:** `src/aios/cli/hrm_hf/device.py`

- [ ] Add XPU case to `resolve_device()`
- [ ] Handle XPU device objects
- [ ] Update strict mode error messages

**Changes:**
```python
def resolve_device(device: str, strict: bool, torch) -> Tuple[str, Any, Any]:
    """Resolve device string and return (dev_str, device_obj, dml_device).

    Handles auto, cuda, xpu, dml with strict mode constraints.
    """
    dev = device
    dml_device = None
    
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            dev = "xpu"
        else:
            try:
                import torch_directml as _dml
                _ = _dml.device()
                dev = "dml"
            except Exception:
                dev = "cpu"
    else:
        # Validate requested device
        if str(dev).lower() == "cuda" and not torch.cuda.is_available():
            # ... existing CUDA error handling ...
        elif str(dev).lower() == "xpu":
            if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
                if strict:
                    from rich import print
                    import typer
                    print({
                        "error": "XPU requested but not available",
                        "hint": "Install intel-extension-for-pytorch or choose --device cpu/cuda/dml",
                        "device_request": "xpu"
                    })
                    raise typer.Exit(code=2)
                else:
                    dev = "cpu"
    
    device_obj = None
    if dev == "dml":
        # ... existing DML handling ...
    else:
        device_obj = torch.device(dev)
    
    return dev, device_obj, dml_device
```

**Tests:**
- [ ] `--device xpu` works with IPEX installed
- [ ] `--device xpu` fails gracefully without IPEX
- [ ] `--device auto` prefers CUDA > XPU > DML > CPU

### Phase 3: Installation Scripts (Medium Priority)

**Estimated Time:** 2 hours

#### Task 3.1: Windows GPU Detection Enhancement
**Files:** `scripts/install_aios_on_windows.ps1`

- [ ] Add AMD GPU detection via WMI
- [ ] Add Intel GPU detection via WMI
- [ ] Offer PyTorch variant selection
- [ ] Add ROCm installation option

**Implementation:**
```powershell
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
      elseif ($vendor -match 'Intel' -or $model -match 'Intel.*Arc|Intel.*Xe') {
        $info.HasIntel = $true
        if ($info.Vendor -eq 'Unknown') { $info.Vendor = 'Intel'; $info.Model = $model }
      }
    }
  } catch {}
  
  try { 
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 
      $info.NvidiaSmi = $true 
    } 
  } catch {}
  
  return $info
}

function Install-PyTorch([string]$pipPath, [string]$pythonPath, [string]$GpuPref) {
  $gpu = $GpuPref
  $gpuInfo = Get-GpuInfo
  
  Write-Host ("[i] GPU detection: Vendor={0} Model={1}" -f $gpuInfo.Vendor, $gpuInfo.Model)
  Write-Host ("[i] GPUs detected: NVIDIA={0} AMD={1} Intel={2}" -f $gpuInfo.HasNvidia, $gpuInfo.HasAmd, $gpuInfo.HasIntel)

  if ($gpu -eq 'auto') {
    if ($gpuInfo.HasNvidia) { 
      $gpu = 'cuda' 
    }
    elseif ($gpuInfo.HasAmd) {
      # ROCm on Windows is experimental, offer DirectML as safer option
      $choice = Read-Host "AMD GPU detected. Use (1) DirectML [Stable] or (2) ROCm [Experimental]? [1/2]"
      if ($choice -eq '2') {
        $gpu = 'rocm'
      } else {
        $gpu = 'dml'
      }
    }
    elseif ($gpuInfo.HasIntel) {
      $gpu = 'xpu'
    }
    else { 
      $gpu = 'cpu' 
    }
  }

  if ($gpu -eq 'cuda') {
    # ... existing CUDA installation ...
  }
  elseif ($gpu -eq 'rocm') {
    Write-Host "[i] Installing PyTorch with ROCm support..."
    Write-Host "[!] Note: ROCm on Windows is experimental" -ForegroundColor Yellow
    try {
      & $pipPath install --upgrade pip wheel setuptools | Out-Null
      # ROCm 6.0 for Windows (if available)
      & $pipPath install torch --index-url https://download.pytorch.org/whl/rocm6.0
      Write-Host "[+] PyTorch installed with ROCm support." -ForegroundColor Green
      return
    } catch {
      Write-Host "[!] ROCm install failed. Falling back to DirectML..." -ForegroundColor Yellow
      $gpu = 'dml'
    }
  }
  elseif ($gpu -eq 'xpu') {
    Write-Host "[i] Installing PyTorch + Intel Extension for PyTorch (XPU)..."
    try {
      & $pipPath install --upgrade pip wheel setuptools | Out-Null
      # CPU PyTorch first
      & $pipPath install torch --index-url https://download.pytorch.org/whl/cpu
      # Then Intel Extension
      & $pipPath install intel-extension-for-pytorch
      
      $code = "import torch, intel_extension_for_pytorch as ipex; print({'torch': torch.__version__, 'ipex': ipex.__version__, 'xpu_available': torch.xpu.is_available()})"
      $res = & $pythonPath -c $code
      Write-Host "[+] PyTorch installed with Intel XPU support. Probe: $res" -ForegroundColor Green
      return
    } catch {
      Write-Host "[!] Intel XPU install failed. Falling back to DirectML..." -ForegroundColor Yellow
      $gpu = 'dml'
    }
  }
  elseif ($gpu -eq 'dml') {
    # ... existing DirectML installation ...
  }

  # ... existing CPU fallback ...
}
```

**Tests:**
- [ ] Test on NVIDIA-only system
- [ ] Test on AMD-only system (if available)
- [ ] Test on Intel-only system (if available)
- [ ] Test GPU preference override (`-Gpu cuda/rocm/xpu/dml/cpu`)

#### Task 3.2: Ubuntu GPU Detection Enhancement
**Files:** `scripts/install_aios_on_ubuntu.sh`

- [ ] Add AMD GPU detection via lspci
- [ ] Add Intel GPU detection via lspci
- [ ] Add ROCm installation option
- [ ] Add IPEX installation option

**Implementation:**
```bash
# GPU Detection
GPU_VENDOR="unknown"
if command -v nvidia-smi >/dev/null 2>&1 || lspci | grep -i nvidia >/dev/null 2>&1; then
  GPU_VENDOR="nvidia"
elif lspci | grep -iE 'amd|radeon' >/dev/null 2>&1; then
  GPU_VENDOR="amd"
elif lspci | grep -iE 'intel.*(arc|xe|iris)' >/dev/null 2>&1; then
  GPU_VENDOR="intel"
fi

echo "[AI-OS] Detected GPU vendor: $GPU_VENDOR"

# PyTorch Installation
case $GPU_VENDOR in
  nvidia)
    echo "[AI-OS] Installing CUDA build of PyTorch (cu121)"
    "$venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
      echo "[AI-OS] CUDA wheel install failed, falling back to CPU build"
      "$venv/bin/pip" install torch torchvision torchaudio
    }
    ;;
  amd)
    echo "[AI-OS] AMD GPU detected. Installing ROCm build of PyTorch"
    echo "[AI-OS] Note: ROCm requires additional system packages"
    # Check if ROCm is installed
    if command -v rocminfo >/dev/null 2>&1; then
      "$venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 || {
        echo "[AI-OS] ROCm wheel install failed, falling back to CPU build"
        "$venv/bin/pip" install torch torchvision torchaudio
      }
    else
      echo "[AI-OS] ROCm not installed on system. Installing CPU build."
      echo "[AI-OS] To use AMD GPU, install ROCm: https://rocm.docs.amd.com/en/latest/"
      "$venv/bin/pip" install torch torchvision torchaudio
    fi
    ;;
  intel)
    echo "[AI-OS] Intel GPU detected. Installing Intel Extension for PyTorch"
    # CPU PyTorch first
    "$venv/bin/pip" install torch torchvision torchaudio
    # Then Intel Extension
    "$venv/bin/pip" install intel-extension-for-pytorch || {
      echo "[AI-OS] Intel Extension install failed, continuing with CPU-only PyTorch"
    }
    ;;
  *)
    echo "[AI-OS] No discrete GPU detected; installing CPU build of PyTorch"
    "$venv/bin/pip" install torch torchvision torchaudio
    ;;
esac
```

**Tests:**
- [ ] Test on Ubuntu with NVIDIA GPU
- [ ] Test on Ubuntu with AMD GPU (if available)
- [ ] Test on Ubuntu with Intel GPU (if available)
- [ ] Test on Ubuntu with no GPU

### Phase 4: GUI Enhancements (Low Priority)

**Estimated Time:** 1 hour

#### Task 4.1: Update Resources Panel Device Labels
**Files:** `src/aios/gui/components/resources_panel/device_management.py`

- [ ] Show vendor in GPU device name
- [ ] Add vendor badges/icons (optional)
- [ ] Color-code by vendor (optional)

**Changes:**
```python
def build_cuda_rows(panel: "ResourcesPanel", devices: list[dict], which: str) -> None:
    """Build GPU selection rows for training or inference."""
    # ... existing code ...
    
    for dev in devices:
        # ... existing parsing ...
        
        name = str(dev.get("name") or f"CUDA {did}")
        vendor = dev.get("vendor", "")
        
        # Add vendor prefix if known and not already in name
        if vendor and vendor not in name:
            display_name = f"[{vendor}] {name}"
        else:
            display_name = name
        
        # ... rest of row building with display_name ...
```

**Tests:**
- [ ] Launch GUI with mixed GPUs
- [ ] Verify vendor labels shown
- [ ] Verify selection still works

#### Task 4.2: Update Status Messages
**Files:** `src/aios/gui/components/resources_panel/device_management.py`

- [ ] Show vendor breakdown in status
- [ ] Differentiate "X NVIDIA GPUs, Y AMD GPUs" vs just "X GPUs"

**Changes:**
```python
def set_detected(panel: "ResourcesPanel", info: dict) -> None:
    # ... existing code ...
    
    # Build vendor summary for status
    vendor_summary = info.get("vendor_summary", {})
    if vendor_summary:
        vendor_parts = [f"{count} {vendor}" for vendor, count in vendor_summary.items()]
        gpu_desc = ", ".join(vendor_parts)
    else:
        gpu_desc = f"{device_count} GPU(s)"
    
    if device_count > 0:
        panel._status_label.config(
            text=f"✓ {gpu_desc} detected",
            foreground="green"
        )
```

**Tests:**
- [ ] Single vendor system shows correct label
- [ ] Mixed vendor system shows breakdown
- [ ] No GPU system shows fallback message

### Phase 5: Documentation (Low Priority)

**Estimated Time:** 1 hour

#### Task 5.1: Update User Documentation

- [ ] Add AMD GPU setup guide to `docs/guide/`
- [ ] Add Intel GPU setup guide to `docs/guide/`
- [ ] Update `README.md` with multi-vendor support claims
- [ ] Add troubleshooting section for GPU detection

**Files to create/update:**
- `docs/guide/AMD_GPU_SETUP.md`
- `docs/guide/INTEL_GPU_SETUP.md`
- `README.md`

#### Task 5.2: Update Developer Documentation

- [ ] Document vendor identification function
- [ ] Add GPU backend architecture notes
- [ ] Update API reference

**Files to create/update:**
- `docs/development/GPU_BACKEND_ARCHITECTURE.md`

### Phase 6: Testing & Validation (Critical)

**Estimated Time:** 2 hours

#### Unit Tests
- [ ] Test `identify_gpu_vendor()` with sample GPU names
- [ ] Test XPU memory detection with mocked torch.xpu
- [ ] Test vendor summary calculation
- [ ] Test device resolution with XPU

**Files:** `tests/test_gpu_detection.py` (new)

#### Integration Tests
- [ ] Test installation script on clean Windows VM
- [ ] Test installation script on clean Ubuntu VM
- [ ] Test GUI with NVIDIA-only system
- [ ] Test training with XPU device (if hardware available)
- [ ] Test torch-info command output

#### Manual Testing Checklist
- [ ] NVIDIA GPU: Detection, labeling, training
- [ ] AMD GPU: Detection, labeling, training (if available)
- [ ] Intel GPU: Detection, labeling, training (if available)
- [ ] Mixed system: All vendors shown correctly (if available)
- [ ] No GPU: Graceful CPU fallback

## Success Criteria

### Phase 1-2 (Core Functionality)
- [x] AMD GPUs correctly labeled with vendor in torch-info
- [x] Intel XPU GPUs detected and shown in GUI
- [x] XPU memory optimization works
- [x] Training works on XPU devices
- [x] No regression for NVIDIA GPU users

### Phase 3-4 (Installation & UX)
- [ ] Installation scripts detect AMD/Intel GPUs
- [ ] Correct PyTorch variant installed automatically
- [ ] GUI shows vendor information
- [ ] Users can distinguish GPU hardware

### Phase 5-6 (Polish & Validation)
- [ ] Documentation complete
- [ ] Tests passing
- [ ] No critical bugs reported

## Known Limitations

### After Implementation
1. **ROCm on Windows** - Experimental, may not work on all AMD GPUs
2. **Intel XPU DDP** - Distributed training requires oneCCL backend (not included)
3. **External GPUs** - Thunderbolt bandwidth limitations (hardware, not software)
4. **Hotplug** - PyTorch may need restart to detect newly connected eGPUs

### Permanent Limitations
1. **Vendor-specific optimizations** - Out of scope, would require per-vendor kernels
2. **Mixed-vendor DDP** - Not possible (requires same backend)
3. **GPU virtualization** - Not tested, may work but unsupported

## Rollback Plan

### If Issues Arise
1. **Vendor detection bugs** - Revert to generic "CUDA" labeling
2. **XPU crashes** - Disable XPU code paths, CPU fallback
3. **Installation issues** - Keep NVIDIA-only detection as fallback

### Breaking Changes
None expected. All changes are additive.

## Performance Impact

Expected performance impact: **None**

- Vendor detection: One-time on startup
- XPU memory checks: Only when XPU devices present
- No changes to training loops

## Security Considerations

None. No security-sensitive changes.

## Dependencies

### New Dependencies
- `intel-extension-for-pytorch>=2.0.0` (optional, hf extra)

### System Requirements
- **AMD GPU users**: ROCm 5.7+ (Linux) or 6.0+ (Windows experimental)
- **Intel GPU users**: Intel GPU drivers, oneAPI Base Toolkit (optional)
- **No change for NVIDIA users**

## Migration Guide

### For Users

**Upgrading from v1.0.x:**

1. Update AI-OS: `git pull && pip install -e .[ui,hf]`
2. AMD GPU users: Reinstall PyTorch with ROCm
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```
3. Intel GPU users: Install IPEX
   ```bash
   pip install intel-extension-for-pytorch
   ```
4. Run `aios torch-info` to verify detection

**No action needed for NVIDIA users** - everything works as before.

### For Developers

**API Changes:**
- `torch_info()` output now includes `vendor` field in device dicts
- `detect_available_vram()` now returns XPU devices
- `_detect_devices_info()` now includes `xpu_devices` key

**New Functions:**
- `identify_gpu_vendor(gpu_name: str, check_rocm: bool) -> str`

## Timeline

### Aggressive (1 week)
- Days 1-2: Phase 1-2 (Detection + XPU)
- Day 3: Phase 3 (Installation scripts)
- Day 4: Phase 4 (GUI)
- Day 5: Phase 5 (Documentation)
- Days 6-7: Phase 6 (Testing)

### Realistic (2 weeks)
- Week 1: Phases 1-3 (Core functionality + installation)
- Week 2: Phases 4-6 (Polish, docs, testing)

### Conservative (3 weeks)
- Week 1: Phases 1-2 (Core functionality)
- Week 2: Phases 3-4 (Installation + GUI)
- Week 3: Phases 5-6 (Documentation + thorough testing)

## Open Questions

1. ~~Should we auto-install ROCm/IPEX or require manual install?~~
   - **Decision:** Auto-detect and offer installation, but don't force
2. ~~Should DirectML remain default for AMD on Windows?~~
   - **Decision:** Yes, ROCm on Windows is experimental
3. ~~How to handle Intel integrated GPUs vs discrete Arc?~~
   - **Decision:** Detect both, let user choose in GUI
4. ~~Version constraints for IPEX?~~
   - **Decision:** `>=2.0.0` for PyTorch 2.x compatibility

## References

### Documentation
- AMD ROCm: https://rocm.docs.amd.com/
- Intel XPU: https://intel.github.io/intel-extension-for-pytorch/
- PyTorch Multi-Backend: https://pytorch.org/docs/stable/notes/hip.html

### Related Issues
- None (proactive enhancement)

### Related PRs
- To be created after implementation

## Approval & Tracking

- [ ] Technical Review - @AI-OS-Team
- [ ] Architecture Review - @AI-OS-Team
- [ ] Implementation Started
- [ ] Phase 1 Complete
- [ ] Phase 2 Complete
- [ ] Phase 3 Complete
- [ ] Phase 4 Complete
- [ ] Phase 5 Complete
- [ ] Phase 6 Complete
- [ ] Documentation Complete
- [ ] Released in v1.1.0

---

**Last Updated:** 2025-10-23  
**Next Review:** After Phase 2 completion  
**Owner:** @AI-OS-Team

## Progress Tracking

### Overall Progress
- [ ] 0% - Not started
- [ ] 25% - Phase 1-2 complete (Core detection + XPU)
- [ ] 50% - Phase 3-4 complete (Installation + GUI)
- [ ] 75% - Phase 5 complete (Documentation)
- [ ] 100% - Phase 6 complete (Testing + Release)

### Current Status
**Status:** 📋 Planned  
**Start Date:** TBD  
**Target Completion:** TBD  
**Actual Completion:** N/A

### Blockers
None currently identified.

### Notes
- Consider coordinating with MIXED_GPU_VENDOR_SUPPORT.md for future mixed-vendor training
- May want to implement basic GPU abstraction layer first for easier future expansion
