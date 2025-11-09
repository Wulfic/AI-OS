## PF-006: Presidio PII redaction for datasets and logs

### Summary

Add optional PII redaction using Microsoft Presidio across dataset ingestion, evaluation samples, and JSONL metrics logging. Expose simple flags and a YAML policy to control which fields and entity types to redact. Provide a small local GUI preview to tune policies.

### Why this matters

- Reduce risk of inadvertently storing PII in training artifacts and logs.
- Help teams comply with stricter data policies without blocking iteration.
- Keep defaults safe and opt-in to minimize performance overhead.

---

## What ships in PF-006

- Utility module: `src/aios/safety/presidio_redactor.py` (Analyzer + Anonymizer pipeline)
- Config file: `config/presidio.yaml` (with `config/presidio.yaml.example` scaffold)
- CLI flags:
	- `--redact-inputs/--no-redact-inputs`
	- `--redact-logs/--no-redact-logs`
	- `--presidio-config PATH`
- Dataset preprocessor CLI: `aios datasets-redact` to create a redacted copy under `training_data/redacted/...`
- Logging hook: optional redaction inside the JSONL logger for sensitive payload fields
- Optional GUI preview (Streamlit/Gradio) to interactively test redaction rules on sample text

---

## Architecture overview

Data paths affected:

1) Ingestion path (training/eval data)
	 - If `--redact-inputs` is enabled, wrap the lines loader and apply redaction per line.
	 - Write redacted datasets with `aios datasets-redact` when needed for offline inspection.

2) Logging path (metrics JSONL)
	 - If `--redact-logs` is enabled, wrap the JSONL writer and redact whitelisted keys: `text`, `sample`, `prompt`, `completion`, `generated`, `context`, and any configured custom keys.

3) Preview GUI (optional)
	 - Small app to paste text, toggle entity types, see the anonymized output and the recognized entities before committing policy changes.

Core components:

- PresidioRedactor (utility): constructs AnalyzerEngine + AnonymizerEngine; exposes `redact_text` and `redact_json`.
- YAML policy: selects entities to target, anonymizer strategies, field-level overrides, and performance knobs.
- Typer flags: propagate redaction options and config path into training/eval CLIs and dataset tools.

---

## Dependencies and setup

Required packages:
- `presidio-analyzer`
- `presidio-anonymizer`
- `spacy`
- spaCy model: `en_core_web_lg` (preferred) or `en_core_web_sm` (lighter, fewer entities)

Windows/PowerShell install (example):

```powershell
# Activate your venv first if needed
# python -m venv .venv; . .\.venv\Scripts\Activate.ps1

pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg
```

Notes:
- For airgapped setups, pre-download wheels and the spaCy model; update `nlp_engine` in YAML accordingly.
- Performance: Presidio adds CPU-bound overhead; consider using `--no-redact-logs` for long runs and enabling only on CI or releases.

---

## Configuration schema (`config/presidio.yaml`)

Minimal keys (all optional; safe defaults apply when file missing):

- `enabled: true|false` — master switch for the utility (CLI flags still gate behavior per path)
- `entities: ["PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", ...]` — target entity types
- `anonymizers:` map of entity → strategy, e.g.:
	- `EMAIL_ADDRESS: { type: "replace", new_value: "<EMAIL>" }`
	- `PHONE_NUMBER: { type: "mask", masking_char: "*", chars_to_mask: 0, from_end: true }`
- `default_anonymizer: { type: "replace", new_value: "<PII>" }`
- `fields:` field-level behavior for JSON/records, e.g.:
	- `text: { redact: true }`
	- `prompt: { redact: true, entities: ["EMAIL_ADDRESS", "PHONE_NUMBER"] }`
	- `user: { redact: false }`
- `custom_recognizers:` list of lightweight recognizers (regex + score + name), optional.
- `nlp_engine:` spaCy model name and language, e.g. `model: en_core_web_lg`, `lang: en`.
- `performance:` batch size, max_workers, timeouts.

See `config/presidio.yaml.example` for a concrete, commented template.

---

## Utility API design (`src/aios/safety/presidio_redactor.py`)

Contract:
- Inputs: `text: str` or `record: dict` with configured fields
- Outputs: redacted text/record plus optional matches metadata
- Error modes: if Presidio is not installed or model missing, log a warning and operate as pass-through (no redaction)

API sketch:
- `class PresidioRedactor:`
	- `__init__(config_path: Optional[str] = None, overrides: Optional[dict] = None)`
	- `redact_text(text: str, *, entities: Optional[list[str]] = None) -> tuple[str, list[dict]]`
	- `redact_json(payload: dict, *, fields: Optional[list[str]] = None) -> tuple[dict, dict]`  # (redacted_copy, matches_by_field)
	- Internal: lazy init AnalyzerEngine/AnonymizerEngine; caching of compiled regex; thread-safe

Implementation notes:
- Use Presidio `AnalyzerEngine` and `AnonymizerEngine` with spaCy NLP engine.
- Apply field-level overrides from YAML (`fields.*`).
- For JSONL payloads, only redact whitelisted keys; never mutate the original dict before writing to disk.

---

## CLI design (Typer)

1) Training CLI: `hrm-hf train-actv1` (`src/aios/cli/hrm_hf_cli.py`)

Add options:
- `--redact-inputs/--no-redact-inputs` (default: no-redact)
- `--redact-logs/--no-redact-logs` (default: no-redact)
- `--presidio-config PATH` (default: `config/presidio.yaml` if exists)

Plumbing:
- Extend `TrainingConfig` with `redact_inputs: bool`, `redact_logs: bool`, `presidio_config: Optional[str]`.
- Pass flags through to `train_actv1_impl` (file: `src/aios/cli/hrm_hf/train_actv1.py`).
- In `train_actv1_impl`, create `PresidioRedactor` early (rank0 only if distributed, broadcast minimal toggles) and:
	- Wrap line ingestion (`get_training_lines`): if `redact_inputs`, map `redact_text` over lines.
	- Wrap `_write_jsonl_helper`: if `redact_logs`, first redact whitelisted payload keys.

2) Datasets CLI: new command `datasets-redact` (`src/aios/cli/datasets_cli.py` -> register)

Command:

```powershell
aios datasets-redact `
	--input training_data/curated_datasets/test_sample.txt `
	--output training_data/redacted/test_sample.txt `
	--presidio-config config/presidio.yaml `
	--format text `
	--json-field text
```

Behavior:
- Reads input (plain text or JSONL), redacts line-by-line, writes output.
- For JSONL, use `--json-field` to pick which key to redact (default: `text`).
- Prints summary with number of lines, redaction hits by entity type.

3) Optional GUI preview: `aios redaction-preview` (future)

- Minimal Streamlit/Gradio app to paste sample text and see redaction live.
- Launch via: `aios redaction-preview --presidio-config config/presidio.yaml`.
- Out of scope for initial PR; spec retained here for future follow-up.

---

## Integration details (where to add hooks)

- File: `src/aios/cli/hrm_hf/train_actv1.py`
	- Initialize redactor once, considering distributed rank.
	- Wrap `_write_jsonl_helper` via local closure `_write_jsonl` to redact payload keys when `config.redact_logs`.
	- During ingestion, before tokenization: if `config.redact_inputs`, map lines through `redact_text`.

- File: `src/aios/cli/hrm_hf/data.py`
	- In `get_training_lines(...)`, inject optional redactor callable to transform lines when `redact_inputs` is enabled.

- File: `src/aios/cli/datasets_cli.py`
	- Register new command `datasets-redact` implemented in `src/aios/cli/datasets/redact_cmd.py`.

Field keys to consider for log redaction (configurable):
- `text`, `sample`, `prompt`, `completion`, `generated`, `context`, `input`, `output`

---

## Testing and acceptance criteria

Unit tests:
- `tests/test_presidio_redaction.py`:
	- Email and phone in a sentence → redacted with placeholders `<EMAIL>`, masked phone.
	- JSON payload: only configured fields are redacted; other keys unchanged.
- `tests/test_logging_redaction.py`:
	- Wrap `_write_jsonl_helper` with redaction; ensure logged file contains redacted content.

Integration checks:
- `aios hrm-hf train-actv1 --redact-logs --log-file artifacts/brains/actv1/metrics.jsonl` produces logs without raw PII.
- `aios datasets-redact --input ... --output ...` writes a redacted dataset copy and prints a summary.

Acceptance:
- When enabled, no raw emails/phones appear in logs or redacted datasets (verified via regex search).
- Redaction disabled by default; enabling adds measurable but acceptable overhead on CPU-only systems.

---

## Risks and mitigations

- False positives/negatives: document scope and provide allow/deny lists; expose `custom_recognizers`.
- Performance overhead: keep redaction opt-in; allow selecting `en_core_web_sm`; support batch processing in the utility.
- Internationalization: default to English model; allow switching NLP model via YAML.

---

## Rollout plan

1) M1 (1 day): Utility + unit tests + example config
	 - Implement `PresidioRedactor` with text and JSON helpers.
	 - Create `config/presidio.yaml.example` and docs.
	 - Add basic unit tests for core behaviors.

2) M2 (1 day): Hooks + CLI + docs
	 - Add CLI flags and wire into training + dataset commands.
	 - Implement `datasets-redact` command.
	 - Expand docs with troubleshooting and examples.

Optional (follow-ups):
- Streamlit/Gradio preview app; VS Code task to launch it.
- Presidio recognizer registry loader from YAML (advanced patterns).

---

## Troubleshooting

- spaCy model error: run `python -m spacy download en_core_web_lg` (or use `en_core_web_sm`).
- Presidio not installed: utility logs a warning and passes text through unchanged.
- Slow runs: disable `--redact-logs` for long training; keep redaction for CI or releases.
- Entity not redacted: verify it’s in `entities` and not excluded by field-level overrides.

---

## Quickstart (Windows/PowerShell)

```powershell
# 1) Install deps
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg

# 2) Copy example config and tweak
Copy-Item config/presidio.yaml.example config/presidio.yaml

# 3) Redact a dataset copy (text)
aios datasets-redact `
	--input training_data/curated_datasets/test_sample.txt `
	--output training_data/redacted/test_sample.txt `
	--presidio-config config/presidio.yaml `
	--format text

# 4) Run training with log redaction
aios hrm-hf train-actv1 `
	--model gpt2 `
	--dataset-file training_data/curated_datasets/test_sample.txt `
	--steps 10 --batch-size 2 `
	--redact-logs `
	--presidio-config config/presidio.yaml `
	--log-file artifacts/brains/actv1/metrics.jsonl
```

---

## Developer checklist (end-to-end)

- [ ] Add `src/aios/safety/presidio_redactor.py` (utility class + lazy init)
- [ ] Extend `TrainingConfig` with `redact_inputs`, `redact_logs`, `presidio_config`
- [ ] Wire flags in `src/aios/cli/hrm_hf_cli.py` and plumb to `train_actv1_impl`
- [ ] In `train_actv1_impl`, wrap `_write_jsonl_helper` with redaction when enabled
- [ ] In `get_training_lines` and/or callsites, map redaction over lines when enabled
- [ ] New CLI `src/aios/cli/datasets/redact_cmd.py` + register in `datasets_cli.py`
- [ ] Add unit tests (`tests/test_presidio_redaction.py`, `tests/test_logging_redaction.py`)
- [ ] Add `config/presidio.yaml.example`
- [ ] Update docs and add Examples section

---

## Operator checklist (before enabling in prod)

- [ ] Verify Presidio + spaCy installed; run the Quickstart to confirm
- [ ] Copy `config/presidio.yaml.example` → `config/presidio.yaml` and tailor entities/placeholders
- [ ] Dry-run `datasets-redact` on a small sample; inspect outputs
- [ ] Enable `--redact-logs` on a short training run; inspect `metrics.jsonl`
- [ ] Monitor performance; consider `en_core_web_sm` if needed

