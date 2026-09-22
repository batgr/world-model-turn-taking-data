# Data pipeline: architecture and onboarding

This directory documents the data pipeline of the conversational world-model
project. Read this page first; the numbered pages describe one capability each,
[`glossary.md`](glossary.md) fixes the vocabulary.

## Core principles

1. Raw data is immutable.
2. Every derived artifact is reproducible from raw data, and every generated
   summary records the code, tools and time that produced it.
3. Detection is separated from correction: audits classify, they never fix.
4. Native timestamps (PTS) are the temporal source of truth; frame or sample
   indices only describe cadence.
5. Dataset-specific semantics are declared in one place per dataset and never
   branch inside generic code.
6. Notebooks explore; `src/` implements; `docs/` states the current contracts;
   `reports/` holds generated evidence.
7. Scientific decisions are documented, with references when possible.
8. Model-specific transformations (resampling, windowing, features) are
   applied as late as possible and are not part of this pipeline yet.

## Where things live

```text
notebooks/   laboratory notebooks: investigation, evidence, exploration
src/conv_wm/ maintained implementation (the only code the CLI and tests run)
docs/        current contracts, terminology and decisions
reports/     generated population evidence (below ${paths.reports}, not versioned)
conf/        configuration (paths, dataset files, manifest options)
tests/       semantics of every reusable module; no test needs the corpus
```

The temporal-media notebook (`notebooks/02_temporal_media_audit.ipynb`) is a
laboratory notebook. It holds the investigations that led to the current
audio taxonomy and is kept as evidence; it is not the implementation and it
is not updated when the implementation changes.

## Stages and capabilities

Data flows `raw → interim → validated → processed → model_ready`. The current
pipeline covers the audits that make raw and interim data trustworthy:

| Capability | Command | Report |
| --- | --- | --- |
| Raw dataset inventory | `conv-wm audit manifest` | `manifest/raw_manifest.parquet`, `manifest/summary.json` |
| Structural validation | `conv-wm audit structure` | `structural/structural_audit.json` |
| Media metadata audit | `conv-wm audit media` | `temporal/media_metadata/` |
| Video timeline audit | `conv-wm audit video` | `temporal/video_timeline/` |
| Audio timeline audit | `conv-wm audit audio` | `temporal/audio_timeline/` |
| A/V technical alignment | `conv-wm audit sync` | `temporal/av_sync/` |
| Annotation integrity | `conv-wm audit annotations` | `annotations/` |
| Targeted annotation cleaning | `conv-wm clean annotations` | `cleaning/annotations/summary.json` |
| Vocal annotation coverage | `conv-wm audit vocal-annotation-coverage --dataset all` | `vocal_annotation_coverage/` |

Report paths are relative to `${paths.reports}` from `conf/config.yaml`. The
media, video, audio, sync and annotation audits read the media metadata table;
media reads the manifest. A missing prerequisite stops the command with the
command that produces it. There is no other ordering constraint.

Label/cue ontology, temporal representation and model-ready packaging are later
capabilities (pages 08–09 remain placeholders).

## One entry point

```bash
uv sync
uv run conv-wm --help
uv run conv-wm audit <capability> [--max-workers N]
uv run conv-wm clean annotations
uv run conv-wm datasets
```

Exit codes: `0` success, `1` the audit could not run (missing artifact or
tool, unreadable input), `2` the audit ran and its verdict is FAIL (structural
or annotation contracts). Every subcommand calls the same function that the
tests exercise (`conv_wm.data.audits.<capability>.run_*_audit(cfg)`); the CLI
adds nothing but argument parsing. The `python -m conv_wm.data.audits.run_*`
modules remain as thin compatibility wrappers.

## Generic versus dataset-specific

```text
                    generic pipeline
        manifest · media · video · audio · sync
        structural validation · annotation integrity
                  annotation cleaning
                          │ reads
          ┌───────────────┼────────────────┐
       EgoCom           Ego4D          <new dataset>
     DatasetSpec      DatasetSpec       DatasetSpec
```

Generic code (`conv_wm.data.manifest`, `conv_wm.data.media.*`,
`conv_wm.data.annotations.*`, `conv_wm.data.audits.*`) knows about files,
containers, streams, tables and annotation semantics — never about a dataset
by name. Everything a dataset contributes is declared in one
`DatasetSpec` (`conv_wm/data/datasets/<name>.py`) and found through the
registry (`conv_wm.data.datasets.get(name)`):

| Part of the spec | Declares | Consumed by |
| --- | --- | --- |
| `structure: StructuralSpec` | tables with Pandera schemas and loaders, relations between them | structural validation |
| `annotations: AnnotationSpec` | annotation sources (scope, time unit and origin, media/entity references, bounds, value fields, provenance, confidence field, known limitations) and cross-source comparisons | annotation integrity |
| `annotation_cleaner` | dataset-specific, explicitly validated minimal transformations | annotation cleaning |
| `audio: AudioInterpretation` | a known boundary grid, extra decoded-validation windows, a dataset summary section | audio timeline audit (relabelling), video timeline audit (extra windows) |

A dataset that is present on disk but not registered is still inventoried and
media-audited with an empty spec.

## Adding a dataset

1. Put the raw files below `${paths.raw}/<Name>/`; the directory name,
   lower-cased, is the dataset key. Add its stage paths and file keys to
   `conf/config.yaml` under `datasets.<key>`.
2. Create `src/conv_wm/data/datasets/<key>.py` with a `DatasetSpec`:
   - `StructuralSpec`: one `TableSpec` per table (a Pandera schema plus a
     `csv_table`/`parquet_table` loader) and the `RelationSpec`s between them;
   - `AnnotationSpec`: one `AnnotationSourceSpec` per annotation table and the
     `CrossSourceComparison`s that make sense;
   - `AudioInterpretation` only if the recordings have known joins or a
     regime worth decoding systematically.
3. Register it in `conv_wm/data/datasets/__init__.py` (built-ins) or from a
   test/fixture with `datasets.register(spec)`.
4. Run `conv-wm audit structure` and `conv-wm audit annotations`; document
   the findings on the capability page.

No generic module changes. `tests/data/test_third_dataset_extensibility.py`
does exactly this with a synthetic affect dataset (video-level mood ratings
and confidence-scored model-inferred intervals) and is the reference example.

## Provenance

Annotation values carry their `provenance` (`human_observed`,
`deterministic_derived`, `transferred`, `pseudo_label`, `cluster_derived`,
`model_inferred`) and, when the source has one, a `confidence_field`. The
audit never infers either; later work that derives or transfers labels must
create new sources with their own provenance rather than overwrite these.

Every summary carries execution provenance (`provenance` key): UTC time,
package version, git commit, dirty-tree flag, Python, ffmpeg and ffprobe
versions, plus the `parameters` that affect interpretation. Reports from
uncommitted code are therefore distinguishable from committed ones.

## Quality checks

```bash
./scripts/fix.sh      # ruff format + autofix
./scripts/check.sh    # ruff format --check, ruff check (incl. D102/D103), pyright, pytest
```

Pyright runs in standard mode over `src/` and `tests/`. Public functions and
methods of `src/` must have docstrings. Notebooks are formatted but exempt
from the blind-except, subprocess-check and docstring rules.
