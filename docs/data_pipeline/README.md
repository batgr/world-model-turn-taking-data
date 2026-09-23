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
             data/pipeline/ the canonical stages; data/vocal/ their pure semantics
             data/datasets/ the dataset adapters; data/audits/ the population audits
docs/        current contracts, terminology and decisions
reports/     generated population evidence (below ${paths.reports}, not versioned)
conf/        configuration (paths, dataset files, manifest options)
tests/       semantics of every reusable module; no test needs the corpus
```

The temporal-media notebook (`notebooks/02_temporal_media_audit.ipynb`) is a
laboratory notebook. It holds the investigations that led to the current
audio taxonomy and is kept as evidence; it is not the implementation and it
is not updated when the implementation changes.

## The v0 pipeline

Data flows `raw → interim → validated → processed → model_ready`. The v0
vocal pipeline is ego-only and annotation-only:

```text
raw annotations
    -> structural / integrity validation      (audits, cleaning)
    -> native focal vocal state               SPEAKING / SILENT / UNKNOWN per wearer
    -> control focal vocal state              the same, requantized at Δ = 100 ms
    -> vocal action grid                      Δ = 100 ms, NO_EVENT / ONSET / OFFSET or masked
    -> model ready                            valid anchors + train/validation/test
```

| Capability | Command | Report |
| --- | --- | --- |
| Raw dataset inventory | `conv-wm audit manifest` | `manifest/raw_manifest.parquet`, `manifest/summary.json` |
| Structural validation | `conv-wm audit structure` | `structural/structural_audit.json` |
| Media metadata audit | `conv-wm audit media` | `temporal/media_metadata/` |
| Video timeline audit | `conv-wm audit video` | `temporal/video_timeline/` |
| Audio timeline audit | `conv-wm audit audio` | `temporal/audio_timeline/` |
| A/V technical alignment | `conv-wm audit sync` | `temporal/av_sync/` |
| Annotation integrity | `conv-wm audit annotations` | `annotations/` |
| Targeted annotation cleaning | `conv-wm clean annotations` | interim tables, `cleaning/annotations/summary.json` |
| **Native focal voice state (v0 layer)** | `conv-wm build native-focal-voice-state --dataset all` | `native_focal_voice_state/<dataset>/`, data in `${paths.processed}/native_focal_voice_state/<dataset>/` |
| **Control focal voice state (v0 layer)** | `conv-wm build control-focal-voice-state --dataset all` | `control_focal_voice_state/<dataset>/`, data in `${paths.processed}/control_focal_voice_state/<dataset>/` |
| **Vocal action grid (v0 layer)** | `conv-wm build vocal-action-grid --dataset all` | `vocal_action_grid/<dataset>/`, data in `${paths.processed}/vocal_action_grid/<dataset>/` |
| **Model ready (v0 layer)** | `conv-wm build model-ready --dataset all` | `model_ready/<dataset>/`, data in `${paths.model_ready}/<dataset>/` |

Report paths are relative to `${paths.reports}` from `conf/config.yaml`. The
media, video, audio, sync and annotation audits read the media metadata table;
media reads the manifest; the native focal voice-state build reads the cleaned
annotation tables, the media metadata table and the manifest; the control focal
voice-state build reads only the native artifact; the vocal action grid reads
the control artifact (and the native timeline its report points at, gridded as
a diagnostic comparison only). Each layer refuses an input whose checksum no
longer matches the report that produced it. A missing prerequisite stops the
command with the command that produces it. There is no other ordering
constraint.

The native state is derived from native annotations only
([`native_focal_voice_state.md`](native_focal_voice_state.md)): Ego4D
camera-wearer `voice_segments` and EgoCom speaker-attributed word timings.
Its two documented limitations — EgoCom absence of annotation is `SILENT`
despite a measured focal-specific coverage of 83.49 %, and Ego4D
`voice_segments` may absorb short internal pauses — are not corrected in v0.

The native timeline is immutable. The control layer
([`control_focal_voice_state.md`](control_focal_voice_state.md)) requantizes it
at Δ = 100 ms with one rule — sub-step silence bridging, `SPEAKING → SILENT
(g < Δ) → SPEAKING` becomes continuous `SPEAKING` — and keeps the provenance of
every bridged gap. Short speech bursts are deliberately not filtered. This is a
statement about the controller's resolution, not a correction of the
annotations.

The vocal action grid ([`vocal_action_grid.md`](vocal_action_grid.md)) samples
the control state every Δ = 100 ms as `NO_EVENT` / `ONSET` / `OFFSET`, masking
any slot the vocabulary cannot express (unknown state or time, more than one
transition) rather than inventing a label. It merges nothing itself and reports
both grids: bridging takes EgoCom's compound slots from 34 495 to 2 083 (valid
action ratio 97.44 % → 99.78 %) and Ego4D's from 20 to 1. What survives is the
short speech bursts, which are kept on purpose.

## Diagnostic audit (not part of v0)

| Capability | Command | Report |
| --- | --- | --- |
| Vocal annotation coverage | `conv-wm audit vocal-annotation-coverage --dataset all` | `vocal_annotation_coverage/` |

The coverage audit runs a voice activity detector on the audio to measure
how much acoustically detected wearer speech the native annotations cover
([`vocal_annotation_coverage.md`](vocal_annotation_coverage.md)). It is
informative — it revealed the EgoCom incompleteness above — and needs the
`coverage-audit` extra; it never modifies v0 and nothing in the v0 pipeline
depends on it beyond the reference recorded in the build report.

The model-ready stage ([`model_ready.md`](model_ready.md)) closes the pipeline:
it indexes the valid training anchors of the grid — enough context, a complete
1 s future, no window crossing a session or a discontinuity — labels each
`event` or `background` from the existing action vocabulary, and assigns
`train` / `validation` / `test` to whole conversations, preferring each
release's own split. Windows are described, never materialized, so a modelling
repository picks the context length at load time. Page 09 remains a
placeholder for model-specific packaging (features, tensors), which belongs to
that repository.

## One entry point

```bash
uv sync
uv run conv-wm --help
uv run conv-wm audit <capability> [--max-workers N]
uv run conv-wm clean annotations
uv run conv-wm build native-focal-voice-state --dataset all
uv run conv-wm build control-focal-voice-state --dataset all
uv run conv-wm build vocal-action-grid --dataset all
uv run conv-wm datasets
```

Exit codes: `0` success, `1` the audit could not run (missing artifact or
tool, unreadable input), `2` the audit ran and its verdict is FAIL (structural
or annotation contracts). Every subcommand calls the same function that the
tests exercise (`conv_wm.data.audits.<capability>.run_*_audit(cfg)` for the
audits, `conv_wm.data.pipeline.<stage>.run_*_build(cfg)` for the build stages);
the CLI adds nothing but argument parsing. `conv-wm build all` runs the build
stages in order and is the canonical entry point.

## Generic versus dataset-specific

```text
                    generic pipeline
        manifest · media · video · audio · sync
        structural validation · annotation integrity
        annotation cleaning · native focal voice state
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
| `native_focal_voice` | cleaned native annotations mapped to one focal voice recording per (view, wearer) | native focal voice-state build |
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
     regime worth decoding systematically;
   - `native_focal_voice` (`<key>_native_voice.py`): the wearer's native
     speech intervals and declared `UNKNOWN` regions per recording, if the
     dataset takes part in the vocal pipeline.
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
