# world-model-turn-taking-data

A reproducible data pipeline that turns egocentric conversation corpora into a
versioned, model-ready turn-taking dataset.

## Overview

Predicting when someone will start or stop speaking needs training data with
exact temporal semantics: who was speaking, when, on which clock, and where the
annotations can be trusted. Raw conversation corpora do not come that way —
each has its own timestamps, its own idea of a "speech segment", and its own
gaps.

This repository takes two such corpora (Ego4D AV and EgoCom) and produces one
dataset with a single contract: a 100 ms grid of the camera wearer's vocal
state and the actions that changed it, plus an index of the training windows
that grid supports. Raw data is never modified, every derived artifact records
what produced it, and every stage refuses an input whose checksum no longer
matches its own report.

**Modelling lives in a separate repository.** This one stops at the dataset.

## Pipeline

```text
raw corpora
   ↓  audit          inventory, structure, media timelines, annotation integrity
   ↓  clean          source-faithful interim tables
   ↓  native_state   annotations   -> SPEAKING / SILENT / UNKNOWN per recording
   ↓  control_state  native state  -> the same, requantized at the 100 ms control step
   ↓  action_grid    control state -> NO_EVENT / ONSET / OFFSET per slot, or masked
   ↓  media_manifest action grid ids -> corpus-relative raw video / audio paths
   ↓  model_ready    action grid   -> valid anchors, splits, dataset card
train / validation / test index  →  a modelling repository
```

| Stage | Why it exists |
| --- | --- |
| **audit** | Decide whether the raw data and its annotations can be trusted at all. Audits classify; they never fix. |
| **clean** | Derive interim tables with explicitly approved transformations, keeping the source's own values. |
| **native_state** | One continuous, exhaustive vocal-state timeline per recording, from native annotations only — no detector, no model. |
| **control_state** | Silences shorter than one control step cannot be represented by a controller running at that step, so they are bridged. A quantization, not a correction. |
| **action_grid** | Sample the state on the regular 100 ms grid and log the one transition a slot may contain. Anything the vocabulary cannot express is masked, never relabelled. |
| **media_manifest** | Resolve every canonical `recording_id` to its raw media, as paths relative to the corpus root, so a consumer never needs a corpus layout. |
| **model_ready** | Index the anchors where a training window can be taken, classify each `event` or `background`, and assign splits to whole conversations. |

Each stage has a page under [`docs/data_pipeline/`](docs/data_pipeline/README.md);
[`glossary.md`](docs/data_pipeline/glossary.md) fixes the vocabulary.

## Repository structure

The repository is `world-model-turn-taking-data`; the Python package it
installs is `conv_wm` and its command is `conv-wm`.

```text
conf/config.yaml                 paths, per-dataset files, manifest options
src/conv_wm/
  cli.py                         the conv-wm command: parsing, config, exit codes
  config.py                      configuration loading and typed stage paths
  provenance.py                  git / ffmpeg / version stamp on every report
  reports.py                     JSON summary and Parquet table writers
  data/
    pipeline/                    the canonical stages, in order
      clean.py  native_state.py  control_state.py  action_grid.py  model_ready.py
    vocal/                       the pure semantics the stages apply (no I/O)
    datasets/                    dataset adapters — the extension point
    audits/                      population audits; they classify, never fix
    media/                       ffprobe access, audio and video timelines
    annotations/                 annotation-source specs and integrity checks
    manifest.py                  raw inventory
    pipeline_inputs.py           read an upstream artifact, refuse a stale one
    validation.py                shared table checks
docs/data_pipeline/              architecture, glossary, one page per stage
notebooks/                       laboratory notebooks (evidence, not implementation)
tests/                           mirrors src/; no test needs the corpus
scripts/                         check.sh and fix.sh
```

## Installation

```bash
git clone <repository-url>
cd world-model-turn-taking-data

uv sync                          # implementation, tests and checks
uv sync --extra notebooks        # plus the exploration libraries
uv sync --extra coverage-audit   # plus Silero VAD, for the diagnostic coverage audit only
```

Python 3.13+. FFmpeg (`ffmpeg` and `ffprobe`) must be on `PATH` for the media
audits; the build stages do not need it.

Without `uv`: `python -m venv .venv && source .venv/bin/activate && pip install -e .`

## Quick start

The build stages read the previous stage's artifact, so the corpus is only
needed for the audits and the cleaning. With a data root in place:

```bash
export EGO_DATA_ROOT=/path/to/your/data      # defaults to ./data

uv run conv-wm audit manifest                # raw inventory
uv run conv-wm audit media                   # container and stream metadata
uv run conv-wm clean annotations             # interim tables
uv run conv-wm build all --dataset all       # the five build stages, in order
uv run conv-wm audit media-manifest          # optional: every media path exists locally
```

`build all` prints the final statistics and the contract checks. To rerun from
one stage on, for instance after changing the control step:

```bash
uv run conv-wm build all --from control-focal-voice-state
```

Individual stages, for development:

```bash
uv run conv-wm build native-focal-voice-state --dataset ego4d
uv run conv-wm build control-focal-voice-state --dataset ego4d
uv run conv-wm build vocal-action-grid --dataset ego4d
uv run conv-wm build media-manifest --dataset ego4d
uv run conv-wm build model-ready --dataset ego4d
```

`uv run conv-wm --help` lists everything. Exit code `1` means a command could
not run (a missing prerequisite names the command that produces it), `2` that a
contract audit reported FAIL.

## Configuration

One mechanism: [`conf/config.yaml`](conf/config.yaml), resolved by OmegaConf,
overridable with `--config`.

| Setting | Where |
| --- | --- |
| data root | `root`, from `$EGO_DATA_ROOT` |
| stage directories | `paths.{raw,interim,validated,processed,model_ready,reports}` |
| per-dataset files | `datasets.<name>.{raw,interim,...}` and `datasets.<name>.files` |
| manifest options | `manifest.compute_checksum`, `manifest.output` |
| VAD / identity settings (diagnostic audit only) | [`conf/vocal_annotation_coverage.yaml`](conf/vocal_annotation_coverage.yaml) |

Window geometry, validity thresholds and the split seed are code-level
defaults, declared once as frozen dataclasses next to the logic that uses them
(`WindowSpec` and `SplitSpec` in `data/pipeline/model_ready.py`, the control
step in `data/vocal/action_grid.py`). They are recorded in every report, so a
build is reproducible from its own artifact.

## Data

**Input.** Whatever the corpus publishes. Ego4D AV expects the v2 annotation
JSONs and the clip videos; EgoCom expects `video_info.csv`,
`ground_truth_transcriptions.csv` and the point-of-view videos. Paths come
from `conf/config.yaml`; nothing is downloaded automatically.

**Canonical internal schema.** After the adapter, every stage speaks the same
language:

| Field | Meaning |
| --- | --- |
| `dataset` | which corpus the row came from |
| `recording_id` | one point-of-view recording — the session |
| `conversation_id` / `sync_group_id` | the conversation several recordings may share |
| `view_id`, `wearer_id` | the video and the person wearing the camera |
| `decision_index`, `decision_time_s` | the 100 ms slot and its time on the recording's canonical clock |
| `focal_state_before` | `SPEAKING` / `SILENT` / `UNKNOWN` just before the slot |
| `action` | `NO_EVENT` / `ONSET` / `OFFSET`, or null when masked |
| `action_valid`, `mask_reason` | whether the slot carries a usable label, and why not |
| `video_path`, `audio_path`, `media_offset_s` | media manifest: the recording's raw files relative to the corpus root, and `media_time_s = decision_time_s + media_offset_s` |

**Dataset adapters.** A dataset contributes a `DatasetSpec`
(`data/datasets/<name>.py`) declaring its tables, annotation semantics,
cleaning rules, the map from its annotations to focal voice recordings, and its
release splits. Everything downstream of the adapter is shared: no core module
branches on a dataset name.

## Outputs

Everything is written below the configured stage roots, outside the repository.

| Artifact | Path |
| --- | --- |
| cleaned annotation tables | `${paths.interim}/<dataset>/` |
| native voice-state timeline | `${paths.processed}/native_focal_voice_state/<dataset>/` |
| control voice-state timeline | `${paths.processed}/control_focal_voice_state/<dataset>/` |
| action grid | `${paths.processed}/vocal_action_grid/<dataset>/vocal_action_grid.parquet` |
| **model-ready index + dataset card** | `${paths.model_ready}/<dataset>/{index.parquet,metadata.json}` |
| **media manifest** | `${paths.model_ready}/<dataset>/media_manifest.parquet` |
| reports for every stage | `${paths.reports}/<stage>/<dataset>/report.json` |

## Model-ready format

`index.parquet` is one row per valid **anchor** — a slot where a training
window can be taken — not a materialized window:

```text
context = [anchor_idx - L + 1, ..., anchor_idx]   L ≤ max_context_steps
future  = [anchor_idx + 1, ..., anchor_idx + future_steps]
```

The anchor belongs to the context; prediction starts at `anchor_idx + 1`. The
consumer picks `L` at load time, so changing the context length needs no
rebuild. Prototype geometry: 10–50 steps of context (1–5 s), a fixed 10-step
horizon (1 s), at 10 Hz.

Each row carries `recording_id`, `conversation_id`, `split`, `anchor_idx`,
`max_context_steps`, `context_valid_ratio`, `future_valid_ratio`,
`sample_class` (`event` / `background`) and `is_trainable`.
[`docs/data_pipeline/model_ready.md`](docs/data_pipeline/model_ready.md) has
the full schema and the reconstruction recipe.

## Reproducibility

The same raw data, config, code revision and seed produce the same artifacts.

- **Config** — one YAML file; its resolved checksum is recorded in every report.
- **Seed** — one: `SplitSpec.seed`, used only when a dataset publishes no split
  of its own. Nothing else in the pipeline is stochastic.
- **Splits** — assigned to whole conversations, never to individual anchors,
  preferring each release's own split; `split_source` records which applied.
- **Ordering** — every stage sorts explicitly before writing; no output depends
  on filesystem iteration order.
- **Lineage** — each report records the input and output SHA-256 checksums, the
  git commit and dirty flag, the exact command, `uv.lock`'s checksum, the
  schema versions and the `lineage_chain`. A stage refuses an input whose
  checksum no longer matches the report that produced it.
- **Dataset card** — `metadata.json` next to the index is self-contained:
  geometry, thresholds, counts per split, contract checks and source checksums.

## Validation and tests

```bash
./scripts/check.sh               # ruff format --check, ruff check, pyright, pytest
./scripts/fix.sh                 # format and autofix

uv run pytest                    # 271 tests, none needs the corpus
uv run pytest tests/test_pipeline_integration.py   # end-to-end on a synthetic fixture
uv run ruff check .
uv run pyright
```

CI runs the same four checks on every push and pull request.

Beyond the test suite, `build model-ready` re-verifies four contracts against
the grid and prints the result: no recording in multiple splits, every anchor
has its minimum context, every anchor has its full future, no anchor crosses a
recording boundary.

## Adding another dataset

1. Write `src/conv_wm/data/datasets/<name>.py` with a `DatasetSpec`: the
   structural schemas, the annotation-source specs, and the optional
   `annotation_cleaner`, `native_focal_voice`, `media_records` and
   `recording_splits` callables.
2. Put the dataset-specific parsing in `<name>_cleaning.py`,
   `<name>_native_voice.py`, `<name>_media.py` and `<name>_splits.py` next to it.
3. Register it in `data/datasets/__init__.py` and add its paths to
   `conf/config.yaml`.
4. Run `conv-wm audit structure`, then `conv-wm build all`.

No core module changes. `tests/data/test_third_dataset_extensibility.py` holds
a synthetic dataset that proves the generic code never branches on a name.

Where code belongs:

```text
dataset-specific parsing   → data/datasets/
pipeline stages            → data/pipeline/
pure semantics, no I/O     → data/vocal/
population audits          → data/audits/
media primitives           → data/media/
```

## Project scope

This repository produces datasets. Model architectures, training loops,
samplers and evaluation live elsewhere and consume `index.parquet`, the
action grid and `media_manifest.parquet`.

## License

No license has been chosen yet; all rights reserved by default. Ego4D and
EgoCom carry their own licences and are not redistributed here.
