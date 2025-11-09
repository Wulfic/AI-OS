# Internationalization and Localization (i18n/L10n)

**Status:** 📋 Planned  
**Priority:** Medium  
**Category:** User Experience & Accessibility  
**Created:** November 9, 2025  
**Target Languages:** Spanish, Portuguese, French, German, Italian, Chinese, Japanese, Arabic, Hindi

---

## Overview

Implement comprehensive internationalization (i18n) and localization (L10n) support to make AI-OS accessible to non-English speaking users worldwide. This includes translating the GUI, CLI, and documentation into 9 additional languages.

**Current State:** English-only hardcoded strings throughout codebase  
**Goal:** Multi-language support with runtime language selection and culturally appropriate formatting

---

## Motivation

### Business Case
- **Market Expansion:** Enable adoption in non-English speaking markets
- **Accessibility:** Remove language barriers for international developers and researchers
- **Community Growth:** Foster global contributor community
- **Competitive Advantage:** Most AI/ML tools remain English-centric

### Target Language Justification
1. **Spanish (es_ES)** - 500M+ speakers, Latin America & Spain markets
2. **Portuguese (pt_BR)** - 250M+ speakers, growing Brazilian tech sector
3. **French (fr_FR)** - 300M+ speakers, European & African markets
4. **German (de_DE)** - 100M+ speakers, strong German engineering community
5. **Italian (it_IT)** - 65M+ speakers, Italian research institutions
6. **Chinese (zh_CN)** - 1.4B+ speakers, massive Chinese AI/ML community
7. **Japanese (ja_JP)** - 125M+ speakers, advanced Japanese tech sector
8. **Arabic (ar_SA)** - 400M+ speakers, growing Middle East tech markets
9. **Hindi (hi_IN)** - 600M+ speakers, booming Indian tech sector

---

## Technical Scope

### Components Requiring Localization

#### 1. GUI (Tkinter Interface)
**Files Affected:** ~50-60 Python files in `src/aios/gui/`

**String Categories:**
- Tab names (Chat, Brains, Datasets, HRM Training, Evaluation, Resources, MCP & Tools, Settings, Debug, Help)
- Button labels (Add Goal, Export CSV, Export JSON, Load Brain, Start Training, etc.)
- Dialog titles and messages (Checkpoint Found, Resume Training, Error, Success, etc.)
- Status messages (Ready, Loading, Processing, Complete, etc.)
- Tooltips and help text
- Error messages and warnings
- Form field labels
- Tree/table column headers
- Menu items

**Estimated String Count:** ~800-1000 unique strings

**Key Files:**
```
src/aios/gui/app/ui_setup.py
src/aios/gui/components/brains_panel/
src/aios/gui/components/chat_panel.py
src/aios/gui/components/datasets_panel/
src/aios/gui/components/hrm_training/
src/aios/gui/components/evaluation_panel/
src/aios/gui/components/resources_panel/
src/aios/gui/components/mcp_panel/
src/aios/gui/components/settings_panel/
src/aios/gui/components/debug_panel.py
src/aios/gui/dialogs/
```

#### 2. CLI (Command-Line Interface)
**Files Affected:** ~15-20 Python files in `src/aios/cli/`

**String Categories:**
- Command descriptions and help text
- Argument/option descriptions
- Error messages
- Success/status messages
- Interactive prompts (in `core_cli.py ui()` function)
- Progress indicators
- Table headers and formatted output

**Estimated String Count:** ~500-700 unique strings

**Key Files:**
```
src/aios/cli/aios.py
src/aios/cli/core_cli.py
src/aios/cli/hrm_cli.py
src/aios/cli/hrm_hf_cli.py
src/aios/cli/eval_cli.py
src/aios/cli/datasets/
src/aios/cli/optimization_cli.py
src/aios/cli/modelcard_cli.py
```

#### 3. Documentation
**Files Affected:** All markdown files in `docs/`

**Content:**
- README.md
- Installation guides
- User guides
- API documentation
- Contributing guidelines

**Estimated Page Count:** ~50-100 documentation pages

**Strategy:** Create separate language subdirectories:
```
docs/
  ├── en/  (English - default)
  ├── es/  (Spanish)
  ├── pt/  (Portuguese)
  ├── fr/  (French)
  ├── de/  (German)
  ├── it/  (Italian)
  ├── zh/  (Chinese)
  ├── ja/  (Japanese)
  ├── ar/  (Arabic)
  └── hi/  (Hindi)
```

---

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1-2)

#### 1.1 Choose i18n Framework
**Recommended:** Python `gettext` + `babel` for compilation

**Rationale:**
- Standard Python i18n solution
- Excellent tooling (pybabel)
- Wide community support
- Works well with both GUI (Tkinter) and CLI (Typer)

**Dependencies to Add:**
```toml
[project.optional-dependencies]
i18n = [
  "babel>=2.14.0",     # i18n utilities and message extraction
  "polib>=1.2.0",      # .po file manipulation library
]
```

#### 1.2 Create Directory Structure
```
src/aios/
  ├── i18n/
  │   ├── __init__.py           # i18n initialization and utilities
  │   ├── locale_manager.py     # Runtime locale management
  │   └── extract.cfg           # Babel extraction configuration
  └── locales/
      ├── en_US/
      │   └── LC_MESSAGES/
      │       ├── gui.po
      │       ├── cli.po
      │       └── messages.po
      ├── es_ES/
      │   └── LC_MESSAGES/
      ├── pt_BR/
      │   └── LC_MESSAGES/
      ├── fr_FR/
      │   └── LC_MESSAGES/
      ├── de_DE/
      │   └── LC_MESSAGES/
      ├── it_IT/
      │   └── LC_MESSAGES/
      ├── zh_CN/
      │   └── LC_MESSAGES/
      ├── ja_JP/
      │   └── LC_MESSAGES/
      ├── ar_SA/
      │   └── LC_MESSAGES/
      └── hi_IN/
          └── LC_MESSAGES/
```

#### 1.3 Create i18n Utilities Module

**File:** `src/aios/i18n/__init__.py`

```python
"""Internationalization support for AI-OS."""

from __future__ import annotations

import gettext
import locale
import os
from pathlib import Path
from typing import Optional

# Default locale
DEFAULT_LOCALE = "en_US"

# Supported locales
SUPPORTED_LOCALES = {
    "en_US": "English (United States)",
    "es_ES": "Español (España)",
    "pt_BR": "Português (Brasil)",
    "fr_FR": "Français (France)",
    "de_DE": "Deutsch (Deutschland)",
    "it_IT": "Italiano (Italia)",
    "zh_CN": "中文 (简体)",
    "ja_JP": "日本語 (日本)",
    "ar_SA": "العربية (السعودية)",
    "hi_IN": "हिन्दी (भारत)",
}

# Global translation function
_translate = None
_current_locale = DEFAULT_LOCALE


def init_i18n(locale_code: Optional[str] = None) -> None:
    """Initialize i18n system with specified locale.
    
    Args:
        locale_code: Locale code (e.g., 'es_ES'). If None, uses system locale.
    """
    global _translate, _current_locale
    
    if locale_code is None:
        # Try to detect system locale
        try:
            sys_locale = locale.getdefaultlocale()[0]
            locale_code = sys_locale if sys_locale in SUPPORTED_LOCALES else DEFAULT_LOCALE
        except Exception:
            locale_code = DEFAULT_LOCALE
    
    # Validate locale
    if locale_code not in SUPPORTED_LOCALES:
        locale_code = DEFAULT_LOCALE
    
    _current_locale = locale_code
    
    # Set up gettext
    locale_dir = Path(__file__).parent.parent / "locales"
    
    try:
        translation = gettext.translation(
            "messages",
            localedir=str(locale_dir),
            languages=[locale_code],
            fallback=True
        )
        _translate = translation.gettext
    except Exception:
        # Fallback to no-op translation
        _translate = lambda s: s


def _(message: str) -> str:
    """Translate a message to the current locale.
    
    Args:
        message: Message to translate
        
    Returns:
        Translated message
    """
    if _translate is None:
        init_i18n()
    return _translate(message)


def get_current_locale() -> str:
    """Get the current locale code."""
    return _current_locale


def get_supported_locales() -> dict[str, str]:
    """Get dict of supported locale codes to display names."""
    return SUPPORTED_LOCALES.copy()


def set_locale(locale_code: str) -> bool:
    """Set the current locale.
    
    Args:
        locale_code: Locale code to set
        
    Returns:
        True if successful, False otherwise
    """
    if locale_code not in SUPPORTED_LOCALES:
        return False
    
    init_i18n(locale_code)
    return True
```

#### 1.4 Create Babel Configuration

**File:** `src/aios/i18n/extract.cfg`

```ini
[python: **.py]
encoding = utf-8

[javascript: **.js]
encoding = utf-8
```

#### 1.5 Create Extraction Script

**File:** `scripts/extract_translations.py`

```python
#!/usr/bin/env python3
"""Extract translatable strings from source code."""

import subprocess
import sys
from pathlib import Path

def main():
    """Extract strings and create .pot template."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "aios"
    locale_dir = src_dir / "locales"
    pot_file = locale_dir / "messages.pot"
    
    # Ensure locale directory exists
    locale_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract strings
    cmd = [
        "pybabel",
        "extract",
        "-F", str(src_dir / "i18n" / "extract.cfg"),
        "-o", str(pot_file),
        "-k", "_",  # Translation function name
        "--project=AI-OS",
        "--version=1.0.0",
        "--copyright-holder=Wulfic",
        str(src_dir),
    ]
    
    print(f"Extracting translatable strings...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Extracted to {pot_file}")
        print(f"  {result.stdout.strip()}")
    else:
        print(f"✗ Extraction failed:")
        print(result.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### 1.6 Create Compilation Script

**File:** `scripts/compile_translations.py`

```python
#!/usr/bin/env python3
"""Compile .po files to .mo files for runtime use."""

import subprocess
import sys
from pathlib import Path

def main():
    """Compile all .po files to .mo files."""
    project_root = Path(__file__).parent.parent
    locale_dir = project_root / "src" / "aios" / "locales"
    
    compiled_count = 0
    error_count = 0
    
    # Find all .po files
    for po_file in locale_dir.rglob("*.po"):
        mo_file = po_file.with_suffix(".mo")
        
        cmd = [
            "pybabel",
            "compile",
            "-i", str(po_file),
            "-o", str(mo_file),
        ]
        
        print(f"Compiling {po_file.relative_to(project_root)}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            compiled_count += 1
        else:
            error_count += 1
            print(f"  ✗ Error: {result.stderr}")
    
    print(f"\n✓ Compiled {compiled_count} translation(s)")
    if error_count > 0:
        print(f"✗ {error_count} error(s)")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### Phase 2: Code Refactoring (Week 3-5)

#### 2.1 GUI Refactoring Strategy

**Pattern:**
```python
# Before
ttk.Label(frame, text="Model:").pack()
ttk.Button(frame, text="Export CSV", command=callback)
self.status_label = ttk.Label(frame, text="Ready")

# After
from aios.i18n import _
ttk.Label(frame, text=_("Model:")).pack()
ttk.Button(frame, text=_("Export CSV"), command=callback)
self.status_label = ttk.Label(frame, text=_("Ready"))
```

**Files to Refactor (Priority Order):**
1. ✅ Main UI structure (`src/aios/gui/app/ui_setup.py`)
2. ✅ Brains panel (`src/aios/gui/components/brains_panel/`)
3. ✅ Datasets panel (`src/aios/gui/components/datasets_panel/`)
4. ✅ HRM Training panel (`src/aios/gui/components/hrm_training/`)
5. ✅ Evaluation panel (`src/aios/gui/components/evaluation_panel/`)
6. ✅ Chat panel (`src/aios/gui/components/chat_panel.py`)
7. ✅ Settings panel (`src/aios/gui/components/settings_panel/`)
8. ✅ All dialogs (`src/aios/gui/dialogs/`)
9. ✅ Status bar and tooltips

**Gotchas:**
- Dynamic strings with formatting: Use `_("Score: {score}").format(score=value)`
- Pluralization: Use `ngettext()` for singular/plural forms
- Tooltips: Extract to separate translation calls

#### 2.2 CLI Refactoring Strategy

**Pattern:**
```python
# Before
@app.command("train")
def train_command(
    model: str = typer.Option(..., "--model", help="Model name or path"),
):
    """Train a model."""
    print("Training started...")

# After
from aios.i18n import _

@app.command("train")
def train_command(
    model: str = typer.Option(..., "--model", help=_("Model name or path")),
):
    """Train a model."""  # Docstrings extracted separately
    print(_("Training started..."))
```

**Special Considerations for CLI:**
- Help text translation affects `--help` output
- Rich formatted output (tables, progress bars)
- Error messages need careful context
- Interactive prompts in `ui()` function

#### 2.3 String Extraction Guidelines

**DO:**
- ✅ Extract user-facing messages
- ✅ Extract button/label text
- ✅ Extract error messages
- ✅ Extract help text
- ✅ Use context comments for ambiguous strings

**DON'T:**
- ❌ Translate log messages (keep English for debugging)
- ❌ Translate internal identifiers
- ❌ Translate file paths or system commands
- ❌ Translate variable names

**Context Comments:**
```python
# Translator comment for clarity
# Translators: This appears in the training progress dialog
label.config(text=_("Training in progress..."))
```

---

### Phase 3: Translation File Generation (Week 6)

#### 3.1 Extract Strings
```bash
python scripts/extract_translations.py
```

This creates `src/aios/locales/messages.pot` template file.

#### 3.2 Initialize Language Files

For each target language:
```bash
pybabel init -i src/aios/locales/messages.pot \
             -d src/aios/locales \
             -l es_ES

pybabel init -i src/aios/locales/messages.pot \
             -d src/aios/locales \
             -l pt_BR

# ... repeat for all languages
```

This creates `.po` files with message IDs (msgid) and empty translations (msgstr).

#### 3.3 Translation File Structure

**Example: `src/aios/locales/es_ES/LC_MESSAGES/messages.po`**

```po
# Spanish translations for AI-OS
# Copyright (C) 2025 Wulfic
# This file is distributed under the same license as the AI-OS package.

msgid ""
msgstr ""
"Project-Id-Version: AI-OS 1.0.0\n"
"Report-Msgid-Bugs-To: \n"
"POT-Creation-Date: 2025-11-09 10:00+0000\n"
"Language: es_ES\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"

#: src/aios/gui/app/ui_setup.py:63
msgid "Chat"
msgstr "Chat"

#: src/aios/gui/app/ui_setup.py:64
msgid "Brains"
msgstr "Cerebros"

#: src/aios/gui/app/ui_setup.py:65
msgid "Datasets"
msgstr "Conjuntos de datos"

#: src/aios/gui/app/ui_setup.py:66
msgid "HRM Training"
msgstr "Entrenamiento HRM"

#: src/aios/gui/components/brains_panel/panel_main.py:123
msgid "Load Brain"
msgstr "Cargar cerebro"

#: src/aios/gui/components/brains_panel/panel_main.py:145
msgid "Model:"
msgstr "Modelo:"

#: src/aios/gui/components/brains_panel/panel_main.py:201
msgid "Ready"
msgstr "Listo"
```

---

### Phase 4: Translation Work (Week 7-18)

#### 4.1 Translation Approaches

**Option A: Professional Translation Service**
- **Cost:** $0.10-0.30 per word × ~10,000 words = $1,000-3,000 per language
- **Timeline:** 1-2 weeks per language
- **Quality:** High, professional
- **Services:** Gengo, OneHourTranslation, Smartling

**Option B: Community Translation**
- **Cost:** Free (contributor time)
- **Timeline:** 2-4 months per language (variable)
- **Quality:** Variable, requires review
- **Platform:** Crowdin, Weblate, or GitHub-based workflow

**Option C: AI-Assisted + Human Review**
- **Cost:** Low ($100-500 per language for review)
- **Timeline:** 2-4 weeks per language
- **Quality:** Good with proper review
- **Process:**
  1. Use GPT-4/Claude for initial translation
  2. Native speaker review and correction
  3. Context validation

**Recommended:** Option C for speed and cost-effectiveness

#### 4.2 Translation Priority Order

**Tier 1 (Weeks 7-10):** Western European languages
1. Spanish (es_ES) - Week 7
2. French (fr_FR) - Week 8
3. Portuguese (pt_BR) - Week 9
4. German (de_DE) - Week 10
5. Italian (it_IT) - Week 10

**Tier 2 (Weeks 11-14):** East Asian languages
6. Chinese Simplified (zh_CN) - Week 11-12
7. Japanese (ja_JP) - Week 13-14

**Tier 3 (Weeks 15-18):** Complex scripts
8. Arabic (ar_SA) - Week 15-16
9. Hindi (hi_IN) - Week 17-18

#### 4.3 Translation Quality Checklist

For each language:
- [ ] All strings translated (no empty msgstr)
- [ ] Technical terminology consistent
- [ ] Proper capitalization and punctuation
- [ ] Formatting placeholders preserved (`{0}`, `%s`, etc.)
- [ ] Plural forms correctly implemented
- [ ] Native speaker reviewed
- [ ] Context-appropriate tone (formal vs. informal)
- [ ] UI tested with translations loaded

#### 4.4 Special Translation Considerations

**German:**
- Compound words are longer (30-40% more space)
- Example: "Training progress" → "Trainingsfortschritt"
- May need to adjust widget widths

**Chinese/Japanese:**
- No spaces between words
- Vertical text support (not needed for this app)
- Font requirements: Need CJK-compatible fonts

**Arabic:**
- Right-to-left (RTL) text direction
- Requires significant UI layout changes
- Numbers may be displayed left-to-right within RTL text
- Consider deferring to later phase

**Hindi:**
- Devanagari script
- Font rendering support needed
- May need line-height adjustments

---

### Phase 5: UI Layout Adjustments (Week 19-20)

#### 5.1 Dynamic Widget Sizing

**Problem:** Different languages have different text lengths

**Solution:** Use dynamic sizing instead of fixed widths

```python
# Before
entry = ttk.Entry(frame, width=20)

# After
entry = ttk.Entry(frame)  # Let it size naturally
entry.pack(fill="x", expand=True)
```

#### 5.2 Text Overflow Handling

**Strategies:**
- Use `wraplength` for labels that might be long
- Add horizontal scrollbars where appropriate
- Increase minimum window size if needed
- Use tooltips for truncated text

```python
label = ttk.Label(
    frame,
    text=_("Very long description text..."),
    wraplength=400  # Wrap at 400 pixels
)
```

#### 5.3 RTL Support (Arabic)

**Challenges:**
- Tkinter has limited RTL support
- May need custom RTL-aware widgets
- Consider using `pack(side="right")` for Arabic layout

**Decision:** Phase 1 implementation will be LTR-only. RTL support deferred to Phase 2.

#### 5.4 Font Support

**Ensure proper fonts installed:**
- Windows: System fonts usually sufficient
- Linux: May need to install language packs
  ```bash
  # For CJK
  sudo apt-get install fonts-noto-cjk
  
  # For Arabic
  sudo apt-get install fonts-noto-nastaliq-urdu
  
  # For Hindi
  sudo apt-get install fonts-noto-devanagari
  ```

---

### Phase 6: Configuration and Runtime Selection (Week 21)

#### 6.1 Add Locale Configuration

**File:** `config/default.yaml`

```yaml
# Internationalization settings
i18n:
  # Default locale (auto-detected if not set)
  locale: null  # Options: en_US, es_ES, pt_BR, fr_FR, de_DE, it_IT, zh_CN, ja_JP, ar_SA, hi_IN
  
  # Fallback locale if selected locale unavailable
  fallback_locale: en_US
```

#### 6.2 GUI Language Selector

Add to Settings panel:

```python
# In src/aios/gui/components/settings_panel/panel_main.py

from aios.i18n import _, get_supported_locales, get_current_locale

class SettingsPanel:
    def __init__(self, ...):
        # ... existing code ...
        
        # Language selection
        lang_frame = ttk.LabelFrame(self, text=_("Language"), padding=10)
        lang_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(lang_frame, text=_("Interface Language:")).pack(anchor="w")
        
        self.locale_var = tk.StringVar(value=get_current_locale())
        locale_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.locale_var,
            values=list(get_supported_locales().keys()),
            state="readonly"
        )
        locale_combo.pack(fill="x", pady=5)
        
        ttk.Button(
            lang_frame,
            text=_("Apply Language (Requires Restart)"),
            command=self._on_language_change
        ).pack()
        
        self.locale_status = ttk.Label(lang_frame, text="", foreground="blue")
        self.locale_status.pack()
    
    def _on_language_change(self):
        """Handle language change."""
        new_locale = self.locale_var.get()
        
        # Save to config
        # ... save logic ...
        
        # Show restart message
        self.locale_status.config(
            text=_("Language will change after restart"),
            foreground="orange"
        )
```

#### 6.3 CLI Language Selection

```bash
# Set via environment variable
export AIOS_LOCALE=es_ES
aios gui

# Or via command line flag
aios --locale es_ES gui

# Or set in config file
aios gui  # Uses config/default.yaml setting
```

---

### Phase 7: Testing and QA (Week 22-24)

#### 7.1 Automated Testing

**Test Coverage:**
- [ ] i18n initialization works for all locales
- [ ] Translation fallback works (missing translations → English)
- [ ] String formatting with placeholders works
- [ ] Plural forms work correctly
- [ ] Language switching doesn't break application

**Test File:** `tests/test_i18n.py`

```python
import pytest
from aios.i18n import init_i18n, _, get_supported_locales

def test_init_default():
    """Test default initialization."""
    init_i18n()
    assert _("Ready") == "Ready"  # English default

def test_init_spanish():
    """Test Spanish initialization."""
    init_i18n("es_ES")
    # Assumes Spanish translation exists
    result = _("Ready")
    assert result == "Listo" or result == "Ready"  # Allow fallback

def test_all_locales_supported():
    """Test all supported locales can be initialized."""
    for locale_code in get_supported_locales():
        init_i18n(locale_code)
        # Should not raise exception

def test_invalid_locale_fallback():
    """Test invalid locale falls back to default."""
    init_i18n("xx_XX")
    assert _("Ready") == "Ready"

def test_formatting():
    """Test string formatting with translations."""
    init_i18n()
    # Assuming translation exists
    msg = _("Score: {score}").format(score=95)
    assert "95" in msg
```

#### 7.2 Manual Testing Checklist

For each language:

**GUI Testing:**
- [ ] All tabs display translated text
- [ ] All buttons have translated labels
- [ ] All dialogs show translated messages
- [ ] Tooltips are translated
- [ ] No text overflow/truncation
- [ ] Status messages update correctly
- [ ] Error messages are clear
- [ ] Help text is accurate

**CLI Testing:**
- [ ] `--help` shows translated text
- [ ] Error messages are translated
- [ ] Interactive prompts are translated
- [ ] Output formatting is correct
- [ ] Progress indicators work

**Functional Testing:**
- [ ] Application functionality unchanged
- [ ] No crashes from translation loading
- [ ] Language switch persists across restarts
- [ ] Fallback to English works if translation missing

#### 7.3 Native Speaker Review

**Requirements:**
- Native speaker fluency
- Technical/ML domain knowledge preferred
- Access to running application

**Review Checklist:**
- [ ] Translation accuracy
- [ ] Natural phrasing (not literal translation)
- [ ] Consistent terminology
- [ ] Appropriate formality level
- [ ] No cultural insensitivity
- [ ] Technical terms correctly used
- [ ] Grammar and spelling correct

---

## Deployment Strategy

### Build Process Updates

**Update:** `pyproject.toml`

```toml
[project.optional-dependencies]
i18n = [
  "babel>=2.14.0",
  "polib>=1.2.0",
]
```

**Update:** Build scripts to compile translations

```bash
# In build process (CI/CD)
python scripts/compile_translations.py
```

### Packaging

**Include compiled .mo files:**
```
src/aios/locales/*/LC_MESSAGES/*.mo
```

**Update MANIFEST.in:**
```
include src/aios/locales/*/LC_MESSAGES/*.mo
```

---

## Maintenance Plan

### Ongoing Translation Updates

**When adding new features:**
1. Use `_()` for all user-facing strings
2. Run `python scripts/extract_translations.py`
3. Update .po files: `pybabel update -i messages.pot -d locales`
4. Translate new strings
5. Compile: `python scripts/compile_translations.py`

### Translation Contributors

**Set up community contribution workflow:**
1. Use Weblate or Crowdin for collaborative translation
2. Or: Accept .po file PRs on GitHub
3. Assign language maintainers for each locale
4. Regular translation reviews (quarterly)

---

## Success Metrics

### Quantitative Metrics
- [ ] 100% of GUI strings translated in all 9 languages
- [ ] 100% of CLI help text translated in all 9 languages
- [ ] 95%+ of documentation translated in priority languages (ES, FR, ZH, JA)
- [ ] < 1% translation-related bug reports
- [ ] Language switching works in < 5 seconds (app restart)

### Qualitative Metrics
- [ ] Native speaker approval rating > 4/5
- [ ] No significant user complaints about translation quality
- [ ] Positive feedback from international user community
- [ ] Increased non-English GitHub issues/discussions

---

## Risks and Mitigation

### Risk 1: Translation Quality
**Risk:** Poor translations create bad user experience  
**Mitigation:** Native speaker review, professional translators for tier 1

### Risk 2: Incomplete Translations
**Risk:** Missing strings show English text  
**Mitigation:** Fallback to English, translation coverage tests

### Risk 3: UI Layout Breaks
**Risk:** Longer text breaks layouts  
**Mitigation:** Dynamic sizing, manual testing, layout guidelines

### Risk 4: Maintenance Burden
**Risk:** Keeping translations updated with new features  
**Mitigation:** Automated extraction, clear contributor guidelines, language maintainers

### Risk 5: RTL Complexity (Arabic)
**Risk:** RTL support is technically complex in Tkinter  
**Mitigation:** Phase 1: LTR-only, Phase 2: RTL investigation/implementation

---

## Alternative Approaches Considered

### 1. English-Only with External Translation Tools
**Pros:** No development work  
**Cons:** Poor UX, unreliable, no control over quality  
**Decision:** Rejected - not professional

### 2. Machine Translation at Runtime
**Pros:** No translation work needed  
**Cons:** Requires internet, latency, poor quality, privacy concerns  
**Decision:** Rejected - unsuitable for professional tool

### 3. Partial Localization (GUI only)
**Pros:** Less work (skip CLI)  
**Cons:** Inconsistent experience  
**Decision:** Considered for MVP, but full coverage preferred

---

## Budget Estimate

### Development Time
- Infrastructure setup: 80 hours
- Code refactoring: 120 hours
- Translation coordination: 40 hours
- Testing and QA: 80 hours
- Documentation: 20 hours
- **Total:** 340 hours

### Translation Costs
**Option A (Professional):**
- 9 languages × $2,000 = $18,000

**Option B (AI + Review):**
- 9 languages × $300 = $2,700

**Option C (Community):**
- Coordinator time: $2,000
- Reviews: $1,000
- **Total:** $3,000

### Recommended Budget
- Development: $17,000 (340 hours @ $50/hr)
- Translation: $3,000 (AI + review)
- **Total:** $20,000

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Infrastructure | 2 weeks | i18n framework, extraction tools |
| 2. Code Refactoring | 3 weeks | All code using `_()` function |
| 3. File Generation | 1 week | .pot and .po files created |
| 4. Translation (Tier 1) | 4 weeks | ES, FR, PT, DE, IT complete |
| 5. Translation (Tier 2) | 4 weeks | ZH, JA complete |
| 6. Translation (Tier 3) | 4 weeks | AR, HI complete |
| 7. UI Adjustments | 2 weeks | Layout fixes, font support |
| 8. Configuration | 1 week | Settings panel, config files |
| 9. Testing & QA | 3 weeks | All locales tested |
| **Total** | **24 weeks** | **Full localization support** |

**Accelerated Timeline:** 16 weeks (with parallel work and larger team)

---

## Dependencies

### Required Libraries
- `babel>=2.14.0` - i18n tooling
- `polib>=1.2.0` - .po file handling
- Font packages for non-Latin scripts (Linux)

### External Dependencies
- Translation service or translators
- Native speaker reviewers
- Testing infrastructure

---

## References

### Standards and Specifications
- [GNU gettext](https://www.gnu.org/software/gettext/)
- [Babel Documentation](http://babel.pocoo.org/)
- [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
- [Unicode CLDR](https://cldr.unicode.org/)

### Best Practices
- [Python i18n/L10n Tutorial](https://docs.python.org/3/library/gettext.html)
- [Tkinter Internationalization](https://tkdocs.com/tutorial/text.html)
- [Translation Best Practices](https://www.w3.org/International/questions/qa-i18n)

---

## Appendix A: Example Translations

### Common UI Strings

| English | Spanish | French | German | Chinese |
|---------|---------|--------|--------|---------|
| Ready | Listo | Prêt | Bereit | 准备就绪 |
| Loading | Cargando | Chargement | Lädt | 加载中 |
| Error | Error | Erreur | Fehler | 错误 |
| Success | Éxito | Succès | Erfolg | 成功 |
| Cancel | Cancelar | Annuler | Abbrechen | 取消 |
| Save | Guardar | Enregistrer | Speichern | 保存 |
| Load Brain | Cargar cerebro | Charger le cerveau | Gehirn laden | 加载大脑 |
| Export CSV | Exportar CSV | Exporter CSV | CSV exportieren | 导出CSV |
| Start Training | Iniciar entrenamiento | Démarrer l'entraînement | Training starten | 开始训练 |

---

## Appendix B: .po File Workflow

### Creating New Language

```bash
# 1. Extract strings
python scripts/extract_translations.py

# 2. Initialize new language
pybabel init -i src/aios/locales/messages.pot \
             -d src/aios/locales \
             -l it_IT

# 3. Edit translations
# Open src/aios/locales/it_IT/LC_MESSAGES/messages.po
# Fill in msgstr values

# 4. Compile
python scripts/compile_translations.py

# 5. Test
AIOS_LOCALE=it_IT aios gui
```

### Updating Existing Language

```bash
# 1. Extract new strings
python scripts/extract_translations.py

# 2. Update .po files
pybabel update -i src/aios/locales/messages.pot \
               -d src/aios/locales

# 3. Translate new strings (marked with "fuzzy")

# 4. Compile
python scripts/compile_translations.py
```

---

## Appendix C: Contribution Guide

### For Translators

**To contribute a translation:**

1. Fork the AI-OS repository
2. Install dependencies: `pip install babel polib`
3. Check if language already initialized:
   - If yes: Update existing .po file
   - If no: Run `pybabel init` for your language
4. Edit `.po` file with a text editor or Poedit
5. Compile to test: `python scripts/compile_translations.py`
6. Test in application: `AIOS_LOCALE=<your_locale> aios gui`
7. Submit pull request with updated .po file

**Translation Guidelines:**
- Keep placeholders like `{0}`, `%s`, `{score}` intact
- Maintain consistent technical terminology
- Use appropriate formality level (usually formal for software)
- Translate meaning, not word-for-word
- Ask for context if unclear

---

**Last Updated:** November 9, 2025  
**Document Version:** 1.0  
**Status:** Ready for Review
