# Training Resume UX Enhancement: Always-Show Checkpoint Screen + Start Position Selector

## Overview

Enhance the GUI “Start Training / Resume” flow so the checkpoint selection screen **always** opens, even for a brand-new run. Add a new option that lets the user choose which **block** and **chunk** to start training from.

This is a UX/planning feature focused on:
- making “resume” behavior explicit and predictable
- enabling controlled partial re-training / targeted continuation
- improving debuggability when chunk tracking state is confusing

## Goals

- Always show the checkpoint/resume screen when the user starts training.
- Provide a simple way to choose a starting point: `block_id` and `chunk_id`.
- Work for both:
  - new runs (no prior checkpoint)
  - resume runs (checkpoint exists)
- Keep behavior consistent across single-GPU and parallel-independent multi-GPU.

## Non-Goals

- Do not add complex browsing/search/filtering of dataset contents.
- Do not add per-sample selection or editing.
- Do not change the adaptive LR auto-mode logic (auto stays limited to built-in modes).

## Proposed UX

### Checkpoint Screen Behavior

When the user clicks “Start Training”:
- Always open the checkpoint screen.
- Screen provides two primary choices:
  1) **Start New Run** (no checkpoint required)
  2) **Resume From Checkpoint** (choose existing checkpoint if available)

### Start Position Selector (New)

Add a section:

- **Start Position**
  - Block: numeric input (integer, >= 0)
  - Chunk: numeric input (integer, >= 0)

Behavior:
- Defaults to `Block=0`, `Chunk=0`.
- Applies to both “Start New Run” and “Resume From Checkpoint”.
- If the chosen block/chunk is out of range (unknown until block discovery), validate as early as possible and fail with a clear message.

### Interaction With Existing Controls

- If the user chooses a start position that conflicts with existing `chunk_tracker_state.json`:
  - prompt to either:
    - respect the tracker (skip already-trained chunks), or
    - force training from the selected position (equivalent to existing `--force-train` escape hatch).

## Backend / Plumbing

### TrainingConfig

Add fields:
- `start_block_id: int = 0`
- `start_chunk_id: int = 0`

Add CLI args (HRM HF):
- `--start-block-id <int>`
- `--start-chunk-id <int>`

These should be independent of `--resume`.

### ChunkTracker / BlockManager Integration

Implement “start position” as the initial cursor for chunk claiming:
- For a new run:
  - start claiming chunks beginning at `(start_block_id, start_chunk_id)`.
- For a resume run:
  - load checkpoint and chunk tracker state as normal
  - then apply the start-position override as the minimum starting point

Important rule:
- The start position is a **lower bound**; chunks earlier than it should not be claimed.

### Parallel-Independent Multi-GPU

When `--parallel-independent` is enabled:
- all workers should share the same start-position lower bound
- distribution logic should skip earlier chunks consistently

## Edge Cases / Validation

- If dataset has fewer blocks/chunks than requested start position:
  - fail fast with a readable error (don’t silently do 0 steps)
- If start position points to an already-completed chunk:
  - behavior depends on “respect tracker” vs “force train” choice
- If user selects Resume but no checkpoint exists:
  - show a warning and allow Start New Run

## Telemetry / Observability

Emit JSONL events at training start:
- `training_start_position_selected` including:
  - `start_block_id`, `start_chunk_id`
  - whether override is active
  - whether `force_train` is active

This makes “why did it train / why did it skip” easy to diagnose.

## Rollout Plan

1) GUI: always show checkpoint screen; add start position inputs.
2) TrainingConfig: add new fields + `to_cli_args()` mapping.
3) HRM HF CLI: add new options + plumb into config.
4) ChunkTracker/Parallel distributor: enforce lower-bound claiming.
5) Add a small diagnostics scenario to verify:
   - new run starts at specified chunk
   - resume run starts at specified chunk
   - parallel-independent respects start position
