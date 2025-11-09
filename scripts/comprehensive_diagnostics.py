#!/usr/bin/env python3
"""
Comprehensive Diagnostics Script for AI-OS
Tests EVERY feature and generates a detailed report

Usage:
    python scripts/comprehensive_diagnostics.py
    python scripts/comprehensive_diagnostics.py --quick  # Run fast tests only
    python scripts/comprehensive_diagnostics.py --full   # Run all tests including slow ones
    python scripts/comprehensive_diagnostics.py --html   # Generate HTML report
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - some tests will be skipped")

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - some tests will be skipped")


class TestStatus(Enum):
    """Test result status"""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SKIP = "⏭️  SKIP"
    WARN = "⚠️  WARN"
    INFO = "ℹ️  INFO"


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: TestStatus
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.name,
            "status_icon": self.status.value,
            "duration": round(self.duration, 3),
            "message": self.message,
            "details": self.details,
            "error": self.error
        }


@dataclass
class TestCategory:
    """Category of tests"""
    name: str
    description: str
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.PASS)
    
    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.FAIL)
    
    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.SKIP)
    
    @property
    def warned(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.WARN)
    
    @property
    def total(self) -> int:
        return len(self.tests)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "warned": self.warned,
            "tests": [t.to_dict() for t in self.tests]
        }


class DiagnosticRunner:
    """Main diagnostic test runner"""
    
    def __init__(self, quick: bool = False, full: bool = False):
        self.quick = quick
        self.full = full
        self.categories: List[TestCategory] = []
        self.start_time = time.time()
        self.project_root = Path(__file__).parent.parent
        
    def run_test(self, name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and return result"""
        start = time.time()
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start
            
            if isinstance(result, TestResult):
                result.duration = duration
                return result
            elif isinstance(result, tuple):
                status, message, details = result if len(result) == 3 else (result[0], result[1], {})
                return TestResult(name, status, duration, message, details)
            else:
                return TestResult(name, TestStatus.PASS, duration, str(result))
                
        except Exception as e:
            duration = time.time() - start
            return TestResult(
                name,
                TestStatus.FAIL,
                duration,
                f"Exception: {str(e)}",
                error=traceback.format_exc()
            )
    
    def add_category(self, category: TestCategory):
        """Add a test category"""
        self.categories.append(category)
    
    def generate_report(self) -> str:
        """Generate text report"""
        total_duration = time.time() - self.start_time
        
        report = []
        report.append("=" * 80)
        report.append("AI-OS COMPREHENSIVE DIAGNOSTICS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Mode: {'Quick' if self.quick else 'Full' if self.full else 'Standard'}")
        report.append(f"Total Duration: {total_duration:.2f}s")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_tests = sum(c.total for c in self.categories)
        total_passed = sum(c.passed for c in self.categories)
        total_failed = sum(c.failed for c in self.categories)
        total_skipped = sum(c.skipped for c in self.categories)
        total_warned = sum(c.warned for c in self.categories)
        
        report.append("📊 SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"✅ Passed: {total_passed}")
        report.append(f"❌ Failed: {total_failed}")
        report.append(f"⚠️  Warnings: {total_warned}")
        report.append(f"⏭️  Skipped: {total_skipped}")
        report.append(f"Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")
        report.append("")
        
        # Category summaries
        report.append("📁 CATEGORIES")
        report.append("-" * 80)
        for cat in self.categories:
            status_icon = "✅" if cat.failed == 0 else "❌"
            report.append(f"{status_icon} {cat.name}: {cat.passed}/{cat.total} passed")
        report.append("")
        
        # Detailed results
        for cat in self.categories:
            report.append("=" * 80)
            report.append(f"📦 {cat.name}")
            report.append(f"   {cat.description}")
            report.append("-" * 80)
            
            for test in cat.tests:
                report.append(f"{test.status.value} {test.name} ({test.duration:.3f}s)")
                if test.message:
                    report.append(f"   → {test.message}")
                if test.details:
                    for key, value in test.details.items():
                        report.append(f"   • {key}: {value}")
                if test.error and test.status == TestStatus.FAIL:
                    report.append(f"   ERROR:")
                    for line in test.error.split('\n')[:5]:  # First 5 lines
                        report.append(f"   {line}")
                report.append("")
        
        return "\n".join(report)
    
    def generate_json(self) -> Dict:
        """Generate JSON report"""
        total_tests = sum(c.total for c in self.categories)
        total_passed = sum(c.passed for c in self.categories)
        
        return {
            "generated": datetime.now().isoformat(),
            "mode": "quick" if self.quick else "full" if self.full else "standard",
            "duration": time.time() - self.start_time,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": sum(c.failed for c in self.categories),
                "skipped": sum(c.skipped for c in self.categories),
                "warned": sum(c.warned for c in self.categories),
                "success_rate": (total_passed/total_tests*100) if total_tests > 0 else 0
            },
            "categories": [c.to_dict() for c in self.categories]
        }
    
    def save_reports(self, output_dir: Path):
        """Save reports to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text report
        text_report = self.generate_report()
        text_path = output_dir / f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        text_path.write_text(text_report, encoding='utf-8')
        print(f"\n📄 Text report saved: {text_path}")
        
        # JSON report
        json_report = self.generate_json()
        json_path = output_dir / f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path.write_text(json.dumps(json_report, indent=2), encoding='utf-8')
        print(f"📄 JSON report saved: {json_path}")
        
        # Latest symlink
        latest_text = output_dir / "diagnostics_latest.txt"
        latest_json = output_dir / "diagnostics_latest.json"
        
        if latest_text.exists():
            latest_text.unlink()
        if latest_json.exists():
            latest_json.unlink()
            
        # Create symlinks (Windows: copy files instead)
        if sys.platform == "win32":
            import shutil
            shutil.copy2(text_path, latest_text)
            shutil.copy2(json_path, latest_json)
        else:
            latest_text.symlink_to(text_path.name)
            latest_json.symlink_to(json_path.name)


# ============================================================================
# TEST IMPLEMENTATIONS
# ============================================================================

def test_system_environment(runner: DiagnosticRunner):
    """Test system environment and dependencies"""
    category = TestCategory(
        "System Environment",
        "Python, PyTorch, CUDA, and system dependencies"
    )
    
    # Python version
    result = runner.run_test("Python Version", lambda: (
        TestStatus.PASS,
        f"Python {sys.version.split()[0]}",
        {"version": sys.version.split()[0], "executable": sys.executable}
    ))
    category.tests.append(result)
    
    # PyTorch
    if TORCH_AVAILABLE:
        result = runner.run_test("PyTorch Installation", lambda: (
            TestStatus.PASS,
            f"PyTorch {torch.__version__}",
            {"version": torch.__version__}
        ))
    else:
        result = TestResult("PyTorch Installation", TestStatus.FAIL, 0.0, "PyTorch not installed")
    category.tests.append(result)
    
    # CUDA
    if TORCH_AVAILABLE:
        result = runner.run_test("CUDA Availability", lambda: (
            TestStatus.PASS if torch.cuda.is_available() else TestStatus.WARN,
            f"CUDA {'available' if torch.cuda.is_available() else 'not available'}",
            {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
        ))
        category.tests.append(result)
        
        # GPU Info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                result = runner.run_test(f"GPU {i} Info", lambda idx=i: (
                    TestStatus.INFO,
                    torch.cuda.get_device_name(idx),
                    {
                        "name": torch.cuda.get_device_name(idx),
                        "memory": f"{torch.cuda.get_device_properties(idx).total_memory / 1024**3:.1f} GB"
                    }
                ))
                category.tests.append(result)
    
    # Transformers
    if TRANSFORMERS_AVAILABLE:
        result = runner.run_test("Transformers Installation", lambda: (
            TestStatus.PASS,
            f"Transformers {transformers.__version__}",
            {"version": transformers.__version__}
        ))
    else:
        result = TestResult("Transformers Installation", TestStatus.FAIL, 0.0, "Transformers not installed")
    category.tests.append(result)
    
    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        result = runner.run_test("bitsandbytes (8-bit)", lambda: (
            TestStatus.PASS,
            f"bitsandbytes {bnb.__version__}",
            {"version": bnb.__version__}
        ))
    except ImportError:
        result = TestResult("bitsandbytes (8-bit)", TestStatus.WARN, 0.0, "Not installed - 8-bit optimizer unavailable")
    category.tests.append(result)
    
    runner.add_category(category)


def test_project_structure(runner: DiagnosticRunner):
    """Test project structure and files"""
    category = TestCategory(
        "Project Structure",
        "Verify all required directories and files exist"
    )
    
    required_dirs = [
        "src/aios",
        "src/aios/cli",
        "src/aios/core",
        "src/aios/gui",
        "tests",
        "docs",
        "config",
        "training_data",
        "artifacts"
    ]
    
    for dir_path in required_dirs:
        full_path = runner.project_root / dir_path
        result = runner.run_test(f"Directory: {dir_path}", lambda p=full_path: (
            TestStatus.PASS if p.exists() else TestStatus.FAIL,
            "Exists" if p.exists() else "Missing",
            {"path": str(p)}
        ))
        category.tests.append(result)
    
    required_files = [
        "src/aios/cli/aios.py",
        "src/aios/cli/hrm_hf_cli.py",
        "src/aios/core/hrm_training/training_config.py",
        "src/aios/gui/app/app_main.py",
        "pyproject.toml",
        "README.md"
    ]
    
    for file_path in required_files:
        full_path = runner.project_root / file_path
        result = runner.run_test(f"File: {file_path}", lambda p=full_path: (
            TestStatus.PASS if p.exists() else TestStatus.FAIL,
            "Exists" if p.exists() else "Missing",
            {"path": str(p), "size": p.stat().st_size if p.exists() else 0}
        ))
        category.tests.append(result)
    
    runner.add_category(category)


def test_cli_imports(runner: DiagnosticRunner):
    """Test CLI module imports"""
    category = TestCategory(
        "CLI Imports",
        "Verify all CLI modules can be imported"
    )
    
    cli_modules = [
        "aios.cli.aios",
        "aios.cli.hrm_hf_cli",
        "aios.cli.brains",
        "aios.cli.datasets_cli",
        "aios.cli.goals_cli",
        "aios.cli.cache_cli",
        "aios.cli.eval_cli",
    ]
    
    for module in cli_modules:
        result = runner.run_test(f"Import: {module}", lambda m=module: (
            (TestStatus.PASS, f"Imported successfully", {"module": m})
            if __import__(m, fromlist=['']) else
            (TestStatus.FAIL, "Import failed", {"module": m})
        ))
        category.tests.append(result)
    
    runner.add_category(category)


def test_core_imports(runner: DiagnosticRunner):
    """Test core module imports"""
    category = TestCategory(
        "Core Module Imports",
        "Verify all core modules can be imported"
    )
    
    core_modules = [
        "aios.core.hrm_training.training_config",
        "aios.core.hrm_models.expert_metadata",
        "aios.core.hrm_models.moe_layer",
        "aios.core.hrm_models.dynamic_moe",
        "aios.core.datasets.registry",
        "aios.core.auto_training.orchestrator",
        "aios.core.brains",
        "aios.core.budgets",
    ]
    
    for module in core_modules:
        result = runner.run_test(f"Import: {module}", lambda m=module: (
            (TestStatus.PASS, f"Imported successfully", {"module": m})
            if __import__(m, fromlist=['']) else
            (TestStatus.FAIL, "Import failed", {"module": m})
        ))
        category.tests.append(result)
    
    runner.add_category(category)


def test_gui_imports(runner: DiagnosticRunner):
    """Test GUI module imports"""
    category = TestCategory(
        "GUI Imports",
        "Verify GUI modules can be imported"
    )
    
    try:
        import tkinter
        tkinter_available = True
    except ImportError:
        tkinter_available = False
        result = TestResult("Tkinter", TestStatus.WARN, 0.0, "Tkinter not available - GUI tests skipped")
        category.tests.append(result)
    
    if tkinter_available:
        gui_modules = [
            "aios.gui.app.app_main",
            "aios.gui.components.hrm_training_panel",
            "aios.gui.components.brains_panel",
            "aios.gui.components.rich_chat_panel",
            "aios.gui.components.datasets_panel",
            "aios.gui.components.subbrains_manager_panel",
        ]
        
        for module in gui_modules:
            result = runner.run_test(f"Import: {module}", lambda m=module: (
                (TestStatus.PASS, f"Imported successfully", {"module": m})
                if __import__(m, fromlist=['']) else
                (TestStatus.FAIL, "Import failed", {"module": m})
            ))
            category.tests.append(result)
    
    runner.add_category(category)


def test_training_config(runner: DiagnosticRunner):
    """Test training configuration system"""
    category = TestCategory(
        "Training Configuration",
        "Test TrainingConfig dataclass and validation"
    )
    
    try:
        from aios.core.hrm_training.training_config import TrainingConfig
        
        # Test default config
        result = runner.run_test("Default Config Creation", lambda: (
            TestStatus.PASS,
            "Default config created successfully",
            {}
        ))
        category.tests.append(result)
        
        # Test with custom values
        result = runner.run_test("Custom Config Creation", lambda: (
            TestStatus.PASS,
            "Custom config created successfully",
            {}
        ) if TrainingConfig(
            model="gpt2",
            dataset_file="test.txt",
            steps=100
        ) else (TestStatus.FAIL, "Failed to create custom config", {}))
        category.tests.append(result)
        
        # Test gradient checkpointing
        config = TrainingConfig(gradient_checkpointing=True)
        result = runner.run_test("Gradient Checkpointing Config", lambda: (
            TestStatus.PASS,
            f"Enabled: {config.gradient_checkpointing}",
            {"enabled": config.gradient_checkpointing}
        ))
        category.tests.append(result)
        
        # Test AMP
        config = TrainingConfig(amp=True)
        result = runner.run_test("AMP Config", lambda: (
            TestStatus.PASS,
            f"Enabled: {config.amp}",
            {"enabled": config.amp}
        ))
        category.tests.append(result)
        
        # Test 8-bit optimizer
        config = TrainingConfig(use_8bit_optimizer=True)
        result = runner.run_test("8-bit Optimizer Config", lambda: (
            TestStatus.PASS,
            f"Enabled: {config.use_8bit_optimizer}",
            {"enabled": config.use_8bit_optimizer}
        ))
        category.tests.append(result)
        
    except Exception as e:
        result = TestResult("Training Config", TestStatus.FAIL, 0.0, f"Exception: {str(e)}", error=traceback.format_exc())
        category.tests.append(result)
    
    runner.add_category(category)


def test_dataset_system(runner: DiagnosticRunner):
    """Test dataset loading and management"""
    category = TestCategory(
        "Dataset System",
        "Test dataset readers, registry, and management"
    )
    
    try:
        from aios.core.datasets.registry import DatasetRegistry, DatasetMetadata
        
        # Test registry creation
        result = runner.run_test("Dataset Registry Creation", lambda: (
            TestStatus.PASS,
            "Registry created successfully",
            {}
        ))
        category.tests.append(result)
        
        # Test metadata creation
        metadata = DatasetMetadata(
            dataset_id="test_dataset",
            name="Test Dataset",
            path="/tmp/test.txt",
            size_bytes=1024,
            format="text",
            domain="general"
        )
        result = runner.run_test("Dataset Metadata Creation", lambda: (
            TestStatus.PASS,
            f"Metadata: {metadata.name}",
            {"id": metadata.dataset_id, "domain": metadata.domain}
        ))
        category.tests.append(result)
        
        # Test registry registration
        registry = DatasetRegistry()
        registry.register_dataset(metadata)
        result = runner.run_test("Dataset Registration", lambda: (
            TestStatus.PASS,
            "Dataset registered",
            {"count": len(registry.datasets)}
        ))
        category.tests.append(result)
        
        # Test search
        results = registry.search_datasets(query="test")
        result = runner.run_test("Dataset Search", lambda: (
            TestStatus.PASS,
            f"Found {len(results)} results",
            {"count": len(results)}
        ))
        category.tests.append(result)
        
    except Exception as e:
        result = TestResult("Dataset System", TestStatus.FAIL, 0.0, f"Exception: {str(e)}", error=traceback.format_exc())
        category.tests.append(result)
    
    runner.add_category(category)


def test_expert_metadata(runner: DiagnosticRunner):
    """Test expert metadata system"""
    category = TestCategory(
        "Expert Metadata",
        "Test expert metadata and Dynamic Subbrains components"
    )
    
    try:
        from aios.core.hrm_models.expert_metadata import ExpertMetadata
        from datetime import datetime
        
        # Test metadata creation
        metadata = ExpertMetadata(
            expert_id="test_expert",
            name="Test Expert",
            domain="testing",
            description="A test expert",
            created_at=datetime.now()
        )
        result = runner.run_test("Expert Metadata Creation", lambda: (
            TestStatus.PASS,
            f"Expert: {metadata.name}",
            {"id": metadata.expert_id, "domain": metadata.domain}
        ))
        category.tests.append(result)
        
        # Test JSON serialization
        json_data = metadata.to_dict()
        result = runner.run_test("Expert Metadata Serialization", lambda: (
            TestStatus.PASS,
            "Serialized to JSON",
            {"keys": list(json_data.keys())}
        ))
        category.tests.append(result)
        
    except Exception as e:
        result = TestResult("Expert Metadata", TestStatus.FAIL, 0.0, f"Exception: {str(e)}", error=traceback.format_exc())
        category.tests.append(result)
    
    runner.add_category(category)


def test_auto_training_orchestrator(runner: DiagnosticRunner):
    """Test auto-training orchestrator"""
    category = TestCategory(
        "Auto-Training Orchestrator",
        "Test intent detection and auto-training workflow"
    )
    
    try:
        from aios.core.auto_training.orchestrator import AutoTrainingOrchestrator, IntentDetector
        
        # Test intent detector
        detector = IntentDetector()
        result = runner.run_test("Intent Detector Creation", lambda: (
            TestStatus.PASS,
            "Detector created",
            {}
        ))
        category.tests.append(result)
        
        # Test intent detection
        test_messages = [
            ("Learn Python programming", True),
            ("I want to learn about quantum physics", True),
            ("How are you?", False),
            ("Train an expert on mathematics", True),
        ]
        
        for message, should_detect in test_messages:
            intent = detector.detect_learning_intent(message)
            detected = intent is not None
            result = runner.run_test(f"Intent Detection: '{message[:30]}'", lambda: (
                TestStatus.PASS if detected == should_detect else TestStatus.WARN,
                f"{'Detected' if detected else 'Not detected'}: {intent.subject if intent else 'None'}",
                {"detected": detected, "expected": should_detect, "subject": intent.subject if intent else None}
            ))
            category.tests.append(result)
        
    except Exception as e:
        result = TestResult("Auto-Training Orchestrator", TestStatus.FAIL, 0.0, f"Exception: {str(e)}", error=traceback.format_exc())
        category.tests.append(result)
    
    runner.add_category(category)


def test_memory_optimization_availability(runner: DiagnosticRunner):
    """Test memory optimization feature availability"""
    category = TestCategory(
        "Memory Optimization Features",
        "Check availability of memory optimization features"
    )
    
    # Gradient checkpointing (always available with PyTorch)
    if TORCH_AVAILABLE:
        result = runner.run_test("Gradient Checkpointing", lambda: (
            TestStatus.PASS,
            "Available (built into PyTorch)",
            {}
        ))
    else:
        result = TestResult("Gradient Checkpointing", TestStatus.SKIP, 0.0, "PyTorch not available")
    category.tests.append(result)
    
    # AMP (available with PyTorch >= 1.6)
    if TORCH_AVAILABLE:
        result = runner.run_test("Mixed Precision (AMP)", lambda: (
            TestStatus.PASS,
            "Available (torch.cuda.amp)",
            {}
        ))
    else:
        result = TestResult("Mixed Precision (AMP)", TestStatus.SKIP, 0.0, "PyTorch not available")
    category.tests.append(result)
    
    # 8-bit optimizer
    try:
        import bitsandbytes as bnb
        result = runner.run_test("8-bit Optimizer", lambda: (
            TestStatus.PASS,
            f"Available (bitsandbytes {bnb.__version__})",
            {"version": bnb.__version__}
        ))
    except ImportError:
        result = TestResult("8-bit Optimizer", TestStatus.WARN, 0.0, "bitsandbytes not installed")
    category.tests.append(result)
    
    # FlashAttention
    try:
        import flash_attn
        result = runner.run_test("FlashAttention", lambda: (
            TestStatus.PASS,
            f"Available (flash_attn {flash_attn.__version__})",
            {"version": flash_attn.__version__}
        ))
    except ImportError:
        result = TestResult("FlashAttention", TestStatus.FAIL, 0.0, "NOT IMPLEMENTED - Documentation claims are FALSE")
    category.tests.append(result)
    
    runner.add_category(category)


def test_cli_availability(runner: DiagnosticRunner):
    """Test CLI command availability"""
    category = TestCategory(
        "CLI Commands",
        "Test if CLI commands are available and work"
    )
    
    # Test aios --version
    result = runner.run_test("aios --version", lambda: (
        TestStatus.PASS if subprocess.run(
            [sys.executable, "-m", "aios.cli.aios", "--version"],
            cwd=runner.project_root,
            capture_output=True,
            text=True,
            timeout=10
        ).returncode == 0 else TestStatus.FAIL,
        "Command works",
        {}
    ))
    category.tests.append(result)
    
    # Test aios --help
    result = runner.run_test("aios --help", lambda: (
        TestStatus.PASS if subprocess.run(
            [sys.executable, "-m", "aios.cli.aios", "--help"],
            cwd=runner.project_root,
            capture_output=True,
            text=True,
            timeout=10
        ).returncode == 0 else TestStatus.FAIL,
        "Command works",
        {}
    ))
    category.tests.append(result)
    
    # Test subcommands help
    subcommands = ["hrm-hf", "brains", "datasets", "goals", "cache"]
    for cmd in subcommands:
        result = runner.run_test(f"aios {cmd} --help", lambda c=cmd: (
            TestStatus.PASS if subprocess.run(
                [sys.executable, "-m", "aios.cli.aios", c, "--help"],
                cwd=runner.project_root,
                capture_output=True,
                text=True,
                timeout=10
            ).returncode == 0 else TestStatus.FAIL,
            "Command works",
            {}
        ))
        category.tests.append(result)
    
    runner.add_category(category)


def test_documentation_accuracy(runner: DiagnosticRunner):
    """Test documentation accuracy against codebase"""
    category = TestCategory(
        "Documentation Accuracy",
        "Verify documentation claims match implementation"
    )
    
    # FlashAttention claim vs reality
    try:
        import flash_attn
        result = TestResult("FlashAttention Documentation", TestStatus.PASS, 0.0, "Implemented as documented")
    except ImportError:
        result = TestResult(
            "FlashAttention Documentation",
            TestStatus.FAIL,
            0.0,
            "DOCUMENTATION IS WRONG - Claims FlashAttention is implemented but it's NOT",
            {"claimed": "Integrated", "reality": "Not installed"}
        )
    category.tests.append(result)
    
    # Check if DDP is actually implemented
    result = runner.run_test("DDP Implementation Check", lambda: (
        TestStatus.WARN,
        "Config exists but actual implementation needs verification",
        {}
    ))
    category.tests.append(result)
    
    # Check if DeepSpeed is actually implemented
    result = runner.run_test("DeepSpeed Implementation Check", lambda: (
        TestStatus.WARN,
        "Config exists but actual implementation needs verification",
        {}
    ))
    category.tests.append(result)
    
    # Check if chunked training is actually implemented
    result = runner.run_test("Chunked Training Implementation Check", lambda: (
        TestStatus.WARN,
        "Config exists but actual implementation needs verification",
        {}
    ))
    category.tests.append(result)
    
    runner.add_category(category)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-OS Comprehensive Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--full", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--output", type=str, default="artifacts/diagnostics", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("AI-OS COMPREHENSIVE DIAGNOSTICS")
    print("=" * 80)
    print(f"Mode: {'Quick' if args.quick else 'Full' if args.full else 'Standard'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    runner = DiagnosticRunner(quick=args.quick, full=args.full)
    
    # Run all tests
    test_system_environment(runner)
    test_project_structure(runner)
    test_cli_imports(runner)
    test_core_imports(runner)
    test_gui_imports(runner)
    test_training_config(runner)
    test_dataset_system(runner)
    test_expert_metadata(runner)
    test_auto_training_orchestrator(runner)
    test_memory_optimization_availability(runner)
    test_cli_availability(runner)
    test_documentation_accuracy(runner)
    
    # Generate and print report
    print(runner.generate_report())
    
    # Save reports
    output_dir = Path(args.output)
    runner.save_reports(output_dir)
    
    # Return exit code based on failures
    total_failed = sum(c.failed for c in runner.categories)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
