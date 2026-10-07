# world-model-turn-taking-data

[![CI](https://img.shields.io/github/actions/workflow/status/batgr/world-model-turn-taking-data/ci.yml?branch=main&label=CI)](https://github.com/batgr/world-model-turn-taking-data/actions/workflows/ci.yml)

A reproducible pipeline that turns egocentric conversation corpora (Ego4D AV,
EgoCom) into one versioned, model-ready turn-taking dataset: a regular grid of
the camera wearer's vocal state, the actions that changed it, and an index of
the training windows the grid supports.

Raw data is never modified, every artifact records what produced it, and every
stage refuses an input whose checksum no longer matches its report. Modelling
lives in the [model repository](https://github.com/batgr/world-model-turn-taking-model).

Releases on Hugging Face (no raw media):
[EgoCom, 10 Hz](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom) ·
[EgoCom, 12.5 Hz](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom-12.5hz) ·
EgoCom + Ego4D (`conversational-dynamics-full`, `-full-12.5hz`, access-controlled).

## Pipeline

```text
raw corpora
   ↓  audit          inventory, structure, media timelines, annotation integrity
   ↓  clean          source-faithful interim tables
   ↓  native_state   annotations   -> SPEAKING / SILENT / UNKNOWN per recording
   ↓  control_state  native state  -> the same, requantized at the decision step Δ
   ↓  action_grid    control state -> NO_EVENT / ONSET / OFFSET per slot, or masked
   ↓  media_manifest action grid   -> corpus-relative raw video / audio paths
   ↓  model_ready    action grid   -> valid anchors, splits, dataset card
   ↓  labels         canonical facts + action grid -> optional label sidecars
```

The [pipeline guide](docs/data_pipeline/README.md) explains each stage, links
one page per stage, and describes how to add a dataset;
[`glossary.md`](docs/data_pipeline/glossary.md) fixes the vocabulary.
Observed vocal events are not controllable robot actions.

## Installation

```bash
uv sync                          # package, tests and checks
uv sync --extra labels-audio     # plus Parselmouth, for the prosody labels only
```

Python 3.13+. `ffmpeg` and `ffprobe` must be on `PATH` for the media audits.

## Usage

```bash
export EGO_DATA_ROOT=/path/to/data           # defaults to ./data

uv run conv-wm audit manifest                # raw inventory
uv run conv-wm audit media                   # container and stream metadata
uv run conv-wm clean annotations             # interim tables
uv run conv-wm build all --dataset all       # every build stage, in order
uv run conv-wm release                       # Hugging Face release directories
```

Every setting lives in [`conf/config.yaml`](conf/config.yaml) (paths, decision
grid, release, label parameters). Override one with `--set`, or a whole file
with `--config`. A 12.5 Hz dataset:

```bash
uv run conv-wm --set grid.decision_step_s=0.08 build all --from control-focal-voice-state
```

Window durations (`grid.min_context_s`, `max_context_s`, `future_s`) are in
seconds and rounded up to whole steps: 10/50/10 steps at 10 Hz, 13/63/13 at
12.5 Hz. `uv run conv-wm --help` lists every command. Exit code `1` means a
command could not run (a missing prerequisite names the command that produces
it), `2` that an audit reported FAIL.

## Outputs

| Artifact | Path |
| --- | --- |
| model-ready index + dataset card | `${paths.model_ready}/<dataset>/{index.parquet,metadata.json}` |
| media manifest | `${paths.model_ready}/<dataset>/media_manifest.parquet` |
| action grid | `${paths.processed}/vocal_action_grid/<dataset>/vocal_action_grid.parquet` |
| label sidecars | `${paths.processed}/labels/<dataset>/` |
| one report per stage | `${paths.reports}/<stage>/<dataset>/report.json` |

`index.parquet` has one row per **anchor**, a slot where a window can be taken:
the context ends at the anchor and the future starts at the next slot. The
consumer picks the context length at load time.
[`model_ready.md`](docs/data_pipeline/model_ready.md) has the schema and the
reconstruction recipe; [`labels.md`](docs/data_pipeline/labels.md) and the
generated [`labels_registry.md`](docs/data_pipeline/labels_registry.md)
describe the 116 labels.

Splits are assigned to whole conversations, preferring each release's own
split. The only seed is the split fallback; nothing else is stochastic.

## Checks

```bash
./scripts/check.sh               # ruff format --check, ruff check, pyright, pytest
./scripts/fix.sh                 # format and autofix
```

No test needs the corpus; real-corpus smoke tests are opt-in. CI runs the same
checks on every push and pull request.

## License

No license has been chosen yet; all rights reserved by default. Ego4D and
EgoCom carry their own licences and are not redistributed here.
