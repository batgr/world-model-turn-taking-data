# Control focal voice state (v0)

## What it is

`conv-wm build control-focal-voice-state` sits between the native focal voice
state and the action grid. It answers one question: **what can a controller
running at Δ = 100 ms actually represent of this timeline?**

```text
native annotation
  -> native focal voice state     SPEAKING / SILENT / UNKNOWN, exact seconds
  -> control focal voice state    the same, requantized at Δ
  -> vocal action grid            NO_EVENT / ONSET / OFFSET, or masked
```

The native timeline is **immutable**. This build reads it, refuses it if its
checksum no longer matches its own report, and writes a separate artifact. No
native Parquet file is ever rewritten.

## The one rule: sub-step silence bridging

```text
SPEAKING -> SILENT during g -> SPEAKING     becomes     SPEAKING continuous
                               g < Δ
```

`decision_step_s = Δ = 0.100`. The condition is **strict**: a gap of exactly Δ
is representable and is kept. Equality is decided with the timestamp tolerance
already used everywhere in this pipeline (`BOUNDARY_TOLERANCE_S = 1 ns`); no
source timestamp is modified, rounded or snapped.

This is **not a correction of the annotation**. The native layer still says a
silence happened there, and `bridged_gaps.parquet` still records each one. It
is a quantization tied to the controller's temporal resolution: below one
control step, the controller has no way to close and reopen the channel, so at
that resolution the state is `SPEAKING`.

Successive micro-gaps chain into a single `SPEAKING` interval in one
deterministic left-to-right pass.

## What is deliberately *not* done

The symmetric rule is **not** applied:

```text
SILENT -> short SPEAKING -> SILENT          stays exactly as annotated
```

Short vocalizations (backchannels, brief acknowledgements) are signal, not
noise, and are never filtered — whatever their duration. When such a burst puts
two transitions inside one slot, the action grid masks that slot as
`compound_transition`, exactly as before. Those remaining compounds are the
honest residue of the vocabulary, and the report counts them by pattern.

Nothing else changes: `UNKNOWN` is never bridged across (both frames of a
bridged gap must be `SPEAKING`), no silence longer than or equal to Δ is
touched, and the timeline's span and total duration are preserved.

## Provenance

Each control interval carries where it came from:

| column | meaning |
| --- | --- |
| `dataset`, `recording_id`, `sync_group_id`, `view_id`, `wearer_id` | identity, carried from the native layer |
| `canonical_start_s`, `canonical_end_s`, `voice_state` | the control timeline itself |
| `source_kind`, `source_annotation_id`, `annotation_schema_version` | native evidence; a bridged interval joins the ids of every native `SPEAKING` it absorbed |
| `source_native_index_first`, `source_native_index_last` | the native intervals this row was built from, by position in the recording's native timeline |
| `transform_kind` | `sub_step_silence_bridge`, or null for a native copy |
| `bridged_gap_count`, `bridged_gap_total_duration_s` | how much silence this interval absorbed |
| `control_state_schema_version`, `native_state_schema_version` | the schemas this row was produced under |

`bridged_gaps.parquet` keeps one row per absorbed gap
(`recording_id`, `native_index`, `gap_start_s`, `gap_end_s`, `gap_duration_s`),
so every bridged silence remains individually addressable.

## Invariants (enforced at build time)

The build fails rather than write a timeline that is not a requantization:

- the control intervals **partition** the native ones in order: consecutive
  index ranges, starting at 0, ending at the last native interval;
- the window (`start`, `end`) and the total duration are unchanged;
- only `SPEAKING` intervals carry a `transform_kind`;
- every bridged gap is recorded in `bridged_gaps.parquet`.

A degenerate native timeline (a gap or an overlap, which the native build never
produces) is copied through unchanged; refusing it stays the action grid's job,
which masks that recording as `invalid_timeline`.

## Run and outputs

```bash
uv run conv-wm build native-focal-voice-state --dataset all   # prerequisite
uv run conv-wm build control-focal-voice-state --dataset all
uv run conv-wm build vocal-action-grid --dataset all
```

`${paths.processed}/control_focal_voice_state/<dataset>/control_focal_voice_intervals.parquet`

`${paths.reports}/control_focal_voice_state/<dataset>/`:

- `summary.parquet` — one row per recording: interval and transition counts
  before and after, bridged gaps and their duration, state durations before
  and after;
- `bridged_gaps.parquet` — one row per bridged gap;
- `report.json` — the transform parameters (`transform_kind`, condition,
  strictness, `decision_step_s`, comparison tolerance), the before/after
  statistics, and the MLOps lineage: `control_state_schema_version`,
  `native_state_schema_version`, input native timeline and report checksums,
  output timeline/summary/bridged-gaps checksums, `config_checksum`,
  `git_commit`, `git_dirty`, command, `created_at`, `uv_lock_checksum`, and the
  `lineage_chain` `native annotation → native focal voice state → control focal
  voice state → vocal action grid`.

The build is deterministic and takes seconds. No MLflow, no W&B: the report
*is* the run record.

## Results (2026-09-23)

| | Ego4D | EgoCom |
| --- | --- | --- |
| recordings | 439 | 175 |
| recordings with a bridged gap | 19 | 172 |
| intervals (native → control) | 27 229 → 27 173 | 197 503 → 122 923 |
| resolved transitions (native → control) | 26 777 → 26 721 | 144 176 → 69 596 |
| **bridged gaps** | **28** | **37 290** |
| bridged duration | 0.66 s | 385.38 s |
| bridged duration (median / q95 / max) | 1 ms / 79 ms / 99 ms | 1 ms / 70 ms / 91 ms |
| bridged intervals | 25 | 12 523 |
| max bridged gaps in one interval | 2 | 42 |
| `SPEAKING` duration | 34 081.41 s → 34 082.07 s | 27 574.34 s → 27 959.72 s |
| `SILENT` duration | 97 607.25 s → 97 606.60 s | 111 127.88 s → 110 742.50 s |
| window duration | 131 688.7 s (unchanged) | 138 779.0 s (unchanged) |

Bridged duration is 0.0005 % of Ego4D's valid time and 0.28 % of EgoCom's. The
two corpora react exactly as their annotation conventions predict: Ego4D's
vocal episodes already absorb short internal pauses upstream, EgoCom's
transcript-derived word intervals expose them.

For EgoCom, 31 023 of the 37 290 bridged gaps are shorter than 10 ms — they are
inter-word boundaries in a timed transcript, not conversational pauses.

See [`vocal_action_grid.md`](vocal_action_grid.md) for what this changes on the
grid.
