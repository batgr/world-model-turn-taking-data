# conv_wm

Reproducible data audits for a multimodal conversational world model built on
egocentric conversation datasets (Ego4D AV, EgoCom, and datasets to come).

The pipeline inventories raw data, validates annotation tables, measures
media timelines from native timestamps, and establishes whether annotation
sources are temporally, referentially and semantically trustworthy — without
modifying any raw data. Read
[`docs/data_pipeline/README.md`](docs/data_pipeline/README.md) for the
architecture and [`docs/data_pipeline/glossary.md`](docs/data_pipeline/glossary.md)
for the vocabulary.

## Setup

```bash
uv sync                       # implementation and checks
uv sync --extra notebooks     # plus the exploration libraries
uv sync --extra vad           # plus Silero VAD for vocal coverage auditing
```

FFmpeg (`ffmpeg`/`ffprobe`) must be on `PATH` for the media audits. Data
locations come from `conf/config.yaml`; set `EGO_DATA_ROOT` to point at your
data root.

## Running audits

One entry point:

```bash
uv run conv-wm --help
uv run conv-wm audit manifest      # raw dataset inventory
uv run conv-wm audit structure     # structural validation of dataset tables
uv run conv-wm audit media         # container and stream metadata (needs manifest)
uv run conv-wm audit video         # video timeline cadence (needs media)
uv run conv-wm audit audio         # audio packet and PCM-vs-PTS timelines (needs media)
uv run conv-wm audit sync          # technical A/V alignment (needs media)
uv run conv-wm audit annotations   # annotation integrity contracts (needs media)
uv run conv-wm audit vocal-annotation-coverage --dataset all  # VAD coverage QC
uv run conv-wm clean annotations   # source-faithful derived annotation tables
uv run conv-wm datasets            # registered datasets
```

Reports are written below `${paths.reports}`; every summary carries its
execution provenance (git commit, tool versions, parameters). Exit code `1`
means the audit could not run, `2` that a contract audit reported FAIL.

## Repository layout

```text
src/conv_wm/           maintained implementation
  cli.py               the conv-wm command
  config.py            configuration loading and typed paths
  provenance.py        report provenance (git, ffmpeg, versions)
  reports.py           summary/table writers
  data/manifest.py     raw inventory
  data/media/          ffprobe access, media metadata, video and audio timelines, A/V alignment
  data/annotations/    annotation-source specifications and integrity checks
  data/audits/         one orchestration module per audit + thin run_* wrappers
  data/datasets/       DatasetSpec registry: ego4d.py, egocom.py
docs/data_pipeline/    architecture, glossary, one page per capability
notebooks/             laboratory notebooks (evidence, not implementation)
tests/                 semantics of every module; no test needs the corpus
```

## Development checks

```bash
./scripts/fix.sh      # format and autofix
./scripts/check.sh    # ruff format --check, ruff check, pyright, pytest
```
