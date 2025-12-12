# Datasets - AI-OS
Generated: December 12, 2025
Purpose: Dataset system, readers, registry, and streaming
Status: Implemented

## Files
- Readers: `src/aios/data/datasets/*.py`
- Registry: `src/aios/core/datasets/registry.py`
- Catalog: `src/aios/core/datasets/catalog.py`
- Streaming: `src/aios/data/streaming_cache.py`, `src/aios/data/stream_manager.py`

## Supported Formats
- Plain text, CSV, archives (tar/zip), directories, JSON, JSONL

## Dataset Registry
- Rich metadata (20+ fields), search, recommendations
- Expert-dataset usage tracking, JSON persistence, local scanning

## Streaming Dataset
- Memory-efficient loading, infinite streaming, shuffling, caching

## CLI

The datasets functionality is exposed as discrete commands under the main `aios` CLI:

### Discovery and capacity
- Show storage usage and cap:
	```powershell
	aios datasets-stats
	```
	Output example: `{ "usage_gb": 0.125, "cap_gb": 15.0 }`

- Set storage capacity cap (persisted to ~/.config/aios/datasets.json):
	```powershell
	aios datasets-set-cap 20
	```
	Output example: `{ "ok": true, "cap_gb": 20.0 }`

- List known datasets within size limit:
	```powershell
	aios datasets-list-known --max-size-gb 10
	```
	Output: JSON array of known items `{ name, url, approx_size_gb, notes }`

### Building datasets (web-assisted)
These commands create datasets under the resolved base directory from `datasets_base_dir()`:
`training_data/curated_datasets/<type>/<dataset_name>/`

- Build text dataset by extracting main readable text from top sites:
	```powershell
	aios datasets-build-text "boats" --max-docs 50 --per-site 10 --search-results 10 --min-chars 400 --progress
	```
	Outputs: `manifest.jsonl` with `{ path, label, url, title, chars, excerpt }` and text files.
	Options: `--allow-ext txt,pdf,doc,docx,rtf,md,html,htm` to restrict downloads; `--file-prefix` to prefix filenames; `--store-dataset` to set dataset folder name; `--overwrite` to replace existing.

- Build websites snapshot dataset (HTML pages):
	```powershell
	aios datasets-build-websites "boats" --max-pages 30 --per-site 10 --search-results 10 --min-bytes 2000 --progress
	```
	Outputs: `pages/*.html` and `manifest.jsonl` with `{ path, url, title, bytes, links }`.

- Build images dataset (perceptual dedup optional):
	```powershell
	aios datasets-build-images "boats" --max-images 100 --per-site 20 --pages-per-site 5 --near-duplicate-threshold 8 --allow-ext jpg,png,webp --progress
	```
	Outputs: image files and `manifest.jsonl` with `{ path, label, source_url, page_url, title, alt }`.

- Build generic raw files dataset (by extension allowlist):
	```powershell
	aios datasets-build-raw "boats" --max-files 50 --per-site 10 --allow-ext pdf,csv,json,txt,zip --progress
	```
	Outputs: `files/*` and `manifest.jsonl` with `{ path, label, source_url, page_url, bytes }`.

- Build videos dataset:
	```powershell
	aios datasets-build-videos "boats" --max-videos 20 --per-site 5 --allow-ext mp4,webm --progress
	```
	Outputs: video files and `manifest.jsonl` with `{ path, label, source_url, page_url, bytes }`.

Notes:
- These commands respect a storage capacity cap and will stop early if the cap would be exceeded.
- Networking is best-effort and may skip pages or files if unavailable or too small.
- Use `--overwrite` to rebuild a dataset folder.

### Base directory resolution
Dataset base directory is chosen in this order:
1) `AIOS_DATASETS_DIR` environment variable
2) Project root detection → `training_data/curated_datasets`
3) Fallback: `~/.local/share/aios/datasets`

Use this to find your outputs. Example (project root):
`training_data/curated_datasets/text/<dataset_name>/manifest.jsonl`

## Inputs
- Web-sourced datasets via CLI builders as shown above
- Local files in supported formats (txt/csv/json/jsonl/archives/directories)

## Outputs
- Organized dataset directories under the base dir
- Manifest files (`manifest.jsonl`) describing items for each dataset type
- Storage cap config at `~/.config/aios/datasets.json`

## Try it: quick local example
Use an existing small text file to validate training pipeline compatibility:
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --halt-max-steps 1 --eval-batches 1 --log-file artifacts/brains/actv1/metrics.jsonl
```
Expected: metrics JSONL created and a brain bundle directory under `artifacts/brains/actv1/`.

## Troubleshooting
- "cap_exceeded": Increase cap via `aios datasets-set-cap <GB>` or delete old datasets
- Permission issues on Windows: run VS Code as a user with write access to dataset dir and HF cache dir
- Empty manifests: Increase `--max-docs`/`--max-pages` or relax `--min-chars`/`--min-bytes`

Related: Tokenizers, Core Training, Dynamic Subbrains/MoE

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)