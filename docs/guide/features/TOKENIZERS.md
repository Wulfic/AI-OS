# Tokenizers
Generated: December 12, 2025
Purpose: Tokenizer support and configuration
Status: Implemented (verification varies per model)

## Files
- `src/aios/core/tokenizers/`

## Supported Tokenizers
- Verified: GPT-2 family (default)
- Likely supported via HuggingFace: Qwen, Mistral, Code Llama, DeepSeek-Coder, StarCoder2, Phi-3, Llama 3 (HF auth may be required)
- Not supported: Vision/multimodal and specialized domain tokenizers

## Configuration
- Tokenizer is resolved from the selected `--model` (HF hub id or local path)
- Examples: `gpt2`, `artifacts/hf_implant/base_model`, `mistralai/Mistral-7B-v0.1`
- Local override: place tokenizer files under `artifacts/hf_implant/tokenizers/` and point `--model` to the matching local model path

## Inputs
- Text data from datasets (txt/jsonl), read by dataset readers; tokenization occurs during training/eval
- Tokenizer model files resolved via HuggingFace AutoTokenizer or local tokenizer.json

## Try it: quick check
Tokenization is engaged implicitly by training:
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --halt-max-steps 1 --eval-batches 1 --log-file artifacts/brains/actv1/metrics.jsonl
```
Expected: pipeline loads GPT-2 tokenizer, logs train/eval steps.

## Notes and edge cases
- HF auth for some models: set `HF_TOKEN` env var if private models are required
- Sequence length: Max sequence governed by model config; adjust via training flags (see Core Training)
- Unicode handling: Non-ASCII text is supported; `--ascii-only` exists on some dataset paths to filter
- Mismatched model/tokenizer: Ensure the model path and tokenizer are compatible to avoid errors

Related: Datasets, Core Training

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)