# AI-OS User Guide - Index
**Last Updated**: October 18, 2025  
**Purpose**: Primary navigation hub for all AI-OS documentation  
**Audience**: Users, developers, and contributors

---

## 📖 Welcome to the AI-OS Guide

This guide provides comprehensive documentation for AI-OS, including feature documentation, testing resources, bug tracking, and API references. Start here to navigate all available documentation.

Note on scope: This guide indexes and verifies ONLY files within the `docs/guide/` folder (and its `api/` subfolder). References to other docs are provided for convenience but are out of scope for this guide’s consistency checks.

---

## 🚀 Quick Start

### New Users
1. **Start here**: [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md) - Learn what AI-OS can do
2. **Then review**: [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md) - Compatibility and caveats
3. **API quick reference**: [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md)

### Developers
1. **Feature planning**: [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md)
2. **API reference**: [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md)
3. **Advanced features**: [Advanced Training Features](features/ADVANCED_FEATURES.md)

### Contributors
1. **Run diagnostics (optional)**: See `scripts/README.md`
2. **Open issues**: https://github.com/Wulfic/AI-OS/issues
3. **Verify features**: [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)

---

## 📚 Core Documentation

### Navigation Hubs
Use these to explore features, combinations, and references:
- [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)
- [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md)
- [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md)

---

### [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)
**Feature Documentation** | **1,402 lines** | **100+ features**

Comprehensive mapping of EVERY feature in AI-OS with implementation status.

**Contents**:
- ✅ CLI Commands (15+)
- ✅ Core Training Features (10+)
- ✅ Memory Optimization (6 features)
- ✅ Model Architecture (6 components)
- ✅ Dataset System (5 features)
- ✅ Tokenizers (15 types)
- ✅ Dynamic Subbrains / MoE (7 components)
- ✅ GUI Features (8 panels)
- ⚠️ Multi-GPU & Distributed (needs verification)
- ✅ Tools & Integrations (10+)

**Use this when**: You want to know what features exist and where they're implemented

---

### Issues and Bug Tracking
Use GitHub Issues to report and track problems:
- https://github.com/Wulfic/AI-OS/issues

---

### [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md)
**Compatibility Guide** | **719 lines** | **30+ combinations**

Documents which features work together and known incompatibilities.

**Verified Combinations**:
- ✅ Gradient Checkpointing + AMP (~60-70% memory reduction)
- ✅ Gradient Checkpointing + AMP + 8-bit (~70-80% memory reduction)
- ✅ Streaming Dataset + Shuffling

**Needs Testing**:
- ❓ DDP + anything (unverified)
- ❓ DeepSpeed + anything (unverified)
- ❓ Chunking + anything (unclear if implemented)

**Use this when**: Planning training configurations or debugging feature interactions

---

### Advanced Features and Notes
See advanced flags, platform notes, and troubleshooting:
- [Advanced Training Features](features/ADVANCED_FEATURES.md)
- [Multi-GPU GUI Guide](features/MULTIGPU_GUI_GUIDE.md)

---

## 🔌 API References

### [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md)
**CLI Reference** | **363 lines**

Quick reference for CLI training API changes and usage.

**Contents**:
- What works vs what's removed
- CLI parameter reference
- Python API examples
- Migration guides

**Use this when**: Writing training scripts or using the CLI

---

### [GUI Tooltips Quick Reference](api/TOOLTIPS_QUICK_REFERENCE.md)
**GUI Documentation** | **174 lines**

Complete reference for all GUI tooltips and interface elements.

**Contents**:
- Preset configurations
- Brain naming conventions
- Custom architecture fields
- Resource settings
- Training controls

**Use this when**: Using the GUI or debugging UI issues

---

## ✅ Verification checklist (maintainers)

Use this quick checklist when adding or updating docs in this folder:
- Each file begins with: Title, and “Last Updated: <Month DD, YYYY>”, “Purpose”, optional “Status”
- Links within `docs/guide/` use relative paths without `docs/` prefix
- Links to parent docs (outside guide) use `../` correctly
- Each core doc ends with a “Back to Guide Index” link
- Dates reflect the latest meaningful change
- Terminology: use consistent emoji status markers (✅ ⚠️ ❌ ❓) and section headers
- If new API docs are added, update `docs/guide/api/README.md`

### Link and anchor conventions (contributors)
- Use kebab-case fragment IDs for custom anchors: `### Example Section {#example-section}`
- Prefer relative links within this folder (no leading `docs/`), e.g., `[Core Training](features/CORE_TRAINING.md)`
- When linking to feature docs from feature docs, use `./Other_Feature.md` for siblings and `../INDEX.MD` for guide root
- End each feature doc with: “Back to Feature Index” and “Back to Guide Index” links
- Keep file names as-is; normalize link text and ensure accurate casing in paths on case-sensitive systems

## 🧪 Testing & Diagnostics

### Diagnostics
See `scripts/README.md` for available diagnostic scripts and how to run them.

---

## 📊 Project Health Metrics

### Implementation Status
- ✅ **Fully Implemented**: 58% of features
- ⚠️ **Partially Implemented**: 23% of features
- ❌ **Not Implemented**: 19% of features

### Test Coverage
- ✅ **System Tests Passing**: 77.6% (52/67 tests)
- ⚠️ **Unit Tests**: Minimal coverage (needs expansion)
- ❌ **Integration Tests**: Not implemented

### Bug Status
- 🔴 **Critical Bugs**: 2 open
- 🟠 **High Priority**: 9 open
- 🟡 **Medium Priority**: 9 open
- 🟢 **Low Priority**: 3 open
- **Total**: 26 tracked bugs

### Documentation Quality
- ✅ **Feature Documentation**: 100+ features documented
- ⚠️ **Accuracy**: Some false claims identified
- ✅ **Bug Tracking**: Comprehensive
- ✅ **Testing**: 67 automated tests

---

## 🎯 Common Workflows

### Testing Changes
1. Run diagnostics: `python scripts/comprehensive_diagnostics.py`
2. Review failures in `artifacts/diagnostics/diagnostics_latest.txt`
3. Fix issues identified
4. Re-run tests to verify
5. Update documentation if behavior changed

### Reporting Bugs
1. Check if bug already tracked in GitHub Issues
2. If new, add to appropriate severity section
3. Include: Description, Impact, Reproduction, Workaround, Fix Required
4. Update bug statistics table
5. Add to action plan if high priority

### Verifying Features
1. Find feature in [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)
2. Check implementation status and file location
3. Look for related tests in diagnostic script
4. Test manually if automated tests don't exist
5. Update status (✅/⚠️/❌) based on results
6. Document findings

### Planning Training
1. Review available features in [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)
2. Check compatibility in [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md)
3. Check for known issues in GitHub Issues
4. Refer to [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md)
5. Start training with verified feature combinations

---

## 🔍 Known Critical Issues

### 1. FlashAttention availability and verification
**Severity**: 🔴 CRITICAL  
**Details**: See [Advanced Training Features](features/ADVANCED_FEATURES.md) for platform notes

- Documentation previously overstated FlashAttention as "fully integrated"
- Current code path attempts FA2 under a user-controlled flag (`use_flash_attn`) and falls back to PyTorch SDPA if unavailable
- Availability depends on GPU and whether `flash_attn` is installed
**Impact**: Environment-dependent behavior; performance claims must be qualified

**Action**: Verify on supported hardware, document behavior and fallback clearly

**Action**: Update documentation to remove false claims

---

### 2. Many Features Unverified
**Severity**: 🟠 HIGH  
**Details**: Refer to feature docs and open issues for verification notes

**Unverified Features**:
- DDP (Multi-GPU) - config exists but not tested
- DeepSpeed ZeRO - config exists but not tested
- Chunked training - unclear if implemented
- Most tokenizers - only GPT-2 verified
- Feature combinations - most untested

**Impact**: Unknown if features work as claimed

**Action**: Systematic testing of all features (2-3 week effort)

---

### 3. Limited Test Coverage
**Severity**: 🟠 HIGH  
**Details**: See repository README and CI for coverage and health

- Only 3 test files found in entire project
- 67 system-level diagnostic tests (77.6% passing)
- No unit tests for core functionality
- No integration tests for feature combinations
- No performance benchmarks

**Impact**: Can't verify changes don't break things

**Action**: Add comprehensive test suite (3-4 week effort)

---

## 🛠️ Contributing

### Before Making Changes
1. ✅ Run comprehensive diagnostics
2. ✅ Review related features in Complete Feature Index
3. ✅ Check for related bugs in Bug Tracker
4. ✅ Verify feature combinations won't break

### After Making Changes
1. ✅ Run comprehensive diagnostics again
2. ✅ Update feature status if behavior changed
3. ✅ Add/update bug tracker entries if issues found
4. ✅ Update feature combination matrix if compatibility changed
5. ✅ Update this guide if new features added

---

## 📖 Additional Documentation

### In Other Locations
These documents are NOT in the guide folder but may be referenced:

- **Main README**: `../../README.md` (project root)
- **Quick Start**: `../QUICK_START.md`
- **Docs Index**: `../INDEX.md`
- **Tests**: `../../tests/` folder

---

## 🆘 Getting Help

### For Users
- Check [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md) for feature documentation
- Refer to [GUI Tooltips Quick Reference](api/TOOLTIPS_QUICK_REFERENCE.md) for UI help

### For Developers
- Use [Advanced Training Features](features/ADVANCED_FEATURES.md) for platform notes
- Reference [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md) for API usage

### For Contributors
- See diagnostics in `scripts/README.md`
- Review feature status: [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)
- Verify combinations: [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md)

---

## 📝 Document Status

| Document | Purpose | Last Updated |
|----------|---------|--------------|
| INDEX.MD (this file) | Guide navigation hub | Oct 18, 2025 |
| features/COMPLETE_FEATURE_INDEX.md | Feature docs index | Oct 18, 2025 |
| features/FEATURE_COMBINATION_MATRIX.md | Compatibility guide | Oct 18, 2025 |
| features/ADVANCED_FEATURES.md | Advanced flags and notes | Oct 18, 2025 |
| api/TRAINING_API_QUICK_REFERENCE.md | CLI quick reference | Oct 18, 2025 |
| api/TOOLTIPS_QUICK_REFERENCE.md | GUI tooltips reference | Oct 18, 2025 |

---

## 🎓 Learning Path

### Beginner: Just Starting with AI-OS
1. Review [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md) basics
2. Check [GUI Tooltips Quick Reference](api/TOOLTIPS_QUICK_REFERENCE.md) for UI

### Intermediate: Using AI-OS for Training
1. Start with [Training API Quick Reference](api/TRAINING_API_QUICK_REFERENCE.md)
2. Plan configurations using [Feature Combination Matrix](features/FEATURE_COMBINATION_MATRIX.md)
3. Reference [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md) for advanced features

### Advanced: Contributing to AI-OS
1. See diagnostics in `scripts/README.md`
2. Reference feature docs and API as needed
3. Keep documentation and links accurate

---

## 📞 Support

For questions, issues, or contributions:
- **Issues**: https://github.com/Wulfic/AI-OS/issues
- **Features**: Review [Complete Feature Index](features/COMPLETE_FEATURE_INDEX.md)
- **Documentation**: Use this Guide Index and feature docs

---

**Last Updated**: January 31, 2025  
**Next Review**: February 28, 2025  
**Maintainer**: AI-OS Documentation Team
