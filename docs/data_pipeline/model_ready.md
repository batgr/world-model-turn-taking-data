# Model ready (v0)

## What it is

`conv-wm build model-ready` closes the data pipeline. It turns the vocal action
grid into a published dataset: the aligned temporal sequences the grid already
holds, plus an **index of valid anchors** saying where a training window can be
taken.

```text
vocal action grid  ->  model-ready index  ->  train / validation / test
                                          ->  Hugging Face-ready dataset
```

Modelling lives in a separate repository. This stage produces no tensors, no
sampler and no window arrays; it produces the artifact such a repository loads.

## Windows are described, never materialized

A window is anchored on a decision slot `t` of the grid:

```text
context = [t - L + 1, ..., t]     L ∈ [min_context_steps, max_context_steps]
future  = [t + 1, ..., t + H]     H = future_steps
```

The index stores `max_context_steps` — the largest `L` this anchor supports —
and lets the consumer pick `L` at load time. Nothing is duplicated: one row per
anchor, no `context_1`, `context_2`, … arrays, so changing the context length
later needs no rebuild.

Prototype geometry, at 100 ms per grid slot (10 Hz):

| | steps | seconds |
| --- | --- | --- |
| minimum context | 10 | 1.0 |
| maximum context | 50 | 5.0 |
| future horizon | 10 | 1.0 |

## Valid anchors

An anchor is in the index when, **inside one segment**:

1. at least `min_context_steps` slots exist up to and including it;
2. the full `future_steps` horizon exists after it;
3. it never crosses a recording boundary, nor a hole inside a recording.

A *segment* is a maximal run of consecutive `decision_index` values of one
recording. The action grid emits contiguous slots per recording, so there is
normally one segment per recording; the split on discontinuity is what makes the
boundary guarantee hold anyway.

Validity is then **measured, not filtered**: every structurally valid anchor is
a row, and `is_trainable` says whether it satisfies the thresholds.

```text
min_context_valid_ratio = 0.90     over the largest context the anchor allows
min_future_valid_ratio  = 1.00     over the full horizon
```

`context_valid_ratio` and `future_valid_ratio` are the fraction of slots with
`action_valid` true, so a consumer can apply its own thresholds without
rebuilding. Anchors rejected by the thresholds stay in the index with their
measured ratios rather than disappearing.

## Event and background

`sample_class` is `event` when the future horizon logs at least one action that
is not `NO_EVENT`, and `background` otherwise. The set of event actions is
derived from the existing action vocabulary
([`vocal_action_grid.md`](vocal_action_grid.md)), never re-listed, so the
`Action` enum stays the single definition.

The field exists **only** so a modelling repository can implement its own
sampling. Nothing is rebalanced or reweighted here: the natural distribution is
preserved.

## Splits

Splits are assigned to whole **conversations**, never to individual
anchors:

- an upstream release split is preferred and never recomputed — Ego4D's
  `clips_clean.split`, EgoCom's `video_info_clean.split`, each declared on its
  `DatasetSpec` as `recording_splits`;
- upstream spellings are normalised onto `train` / `validation` / `test`
  (`val` becomes `validation`);
- recordings no upstream split covers are assigned deterministically from
  `SplitSpec.seed` (default 0, 80/10/10), at group level so every
  point-of-view video of one conversation lands in the same split;
- `split_source` records which of the two produced each row.

A split absent upstream stays absent: Ego4D's AV benchmark publishes `train`
and `val` only, so its model-ready dataset has no test split rather than a
synthesised one.

`split_leakage` reports any conversation whose recordings do not share one
split; it is reported, never silently repaired.

## Output

```text
${paths.model_ready}/<dataset>/
├── index.parquet           one row per valid anchor
├── metadata.json           the dataset card: geometry, thresholds, counts, lineage
└── media_manifest.parquet  written by `build media-manifest` (media_manifest.md)
```

When a media manifest built from the same grid exists, `metadata.json` names it
in `files.media_manifest` and describes it in a path-free `media` section; one
built from another grid is refused as stale.

The aligned sequences are the existing grid artifact
(`${paths.processed}/vocal_action_grid/<dataset>/vocal_action_grid.parquet`);
`metadata.json` points at it with its checksum. One `index.parquet` with a
`split` column is enough — no per-split copies are written.

### Index schema

| column | meaning |
| --- | --- |
| `sample_id` | `"<recording_id>#<anchor_idx>"`, unique |
| `dataset`, `recording_id`, `conversation_id`, `view_id`, `wearer_id` | identity; `recording_id` is one point-of-view recording; `conversation_id` groups the recordings of one conversation (`sync_group_id` when the dataset has one) |
| `split`, `split_source` | `train` / `validation` / `test`, and whether it came from upstream or the deterministic fallback |
| `segment_id` | which contiguous run of the recording the anchor is in |
| `anchor_idx`, `anchor_time` | the grid's `decision_index` and `decision_time_s` of the anchor |
| `anchor_row` | the anchor's row position in the grid sorted by `(recording_id, decision_index)` |
| `max_context_steps` | `min(available past+current steps, 50)` |
| `future_steps` | fixed at 10 |
| `context_valid_ratio`, `future_valid_ratio` | measured validity of the two windows |
| `future_event_count` | actions other than `NO_EVENT` in the horizon |
| `sample_class` | `event` or `background` |
| `is_trainable` | both ratios satisfy the thresholds |
| `window_schema_version`, `action_schema_version` | the schemas this row was produced under |

### Reconstructing a window

```text
1. read the sequences table (the vocal action grid);
2. keep the rows whose recording_id == index.recording_id;
3. order them by decision_index;
4. the anchor is the row with decision_index == anchor_idx;
5. context = the L rows ending there, L ≤ max_context_steps;
6. future  = the next future_steps rows.
```

`anchor_row` gives the same position directly when the sequences are read in
the canonical `(recording_id, decision_index)` order.

## Run and outputs

```bash
uv run conv-wm build vocal-action-grid --dataset all   # prerequisite
uv run conv-wm build model-ready --dataset all
```

The build refuses an action grid whose checksum no longer matches its own
report, and prints per dataset the recording and anchor counts per split, the
event/background counts and ratio, the supported context range, the horizon,
the grid frequency, and four contract checks re-verified against the grid:

```text
no recording in multiple splits
every anchor has minimum context
every anchor has full future
no anchor crosses a recording boundary
```

`${paths.reports}/model_ready/<dataset>/` holds `summary.parquet` (one row per
recording) and `report.json` (the statistics plus the usual lineage:
checksums of the input grid and the output artifacts, `config_checksum`,
`git_commit`, `git_dirty`, command, `created_at`, `uv_lock_checksum` and the
`lineage_chain`).

## Results (2026-09-23)

| | Ego4D | EgoCom |
| --- | --- | --- |
| recordings | 439 | 175 |
| anchors | 1 308 469 | 1 384 465 |
| rejected by validity | 11 | 21 509 |
| **event / background** | 242 428 / 1 066 030 | 413 350 / 949 606 |
| event/background ratio | 0.2274 | 0.4353 |
| train (recordings / anchors) | 389 / 1 159 418 | 131 / 1 072 231 |
| validation (recordings / anchors) | 50 / 149 040 | 18 / 85 003 |
| test (recordings / anchors) | — (no upstream test split) | 26 / 205 722 |
| split source | upstream | upstream |

EgoCom rejects more anchors because its `UNKNOWN` media regions make
`future_valid_ratio < 1`; Ego4D's 11 rejections are the anchors whose context
still reaches a recording's masked first slot.

## Assumptions

- **No multimodal feature column exists yet.** The v0 grid is vocal and
  annotation-only, so a window's per-step observation is its
  `focal_state_before` / `action` sequence. `WindowSpec.feature_columns` is the
  hook for later feature columns and is empty.
- **`recording_id` is the grid's `recording_id`** — one point-of-view recording.
  `conversation_id` is the conversation, which is what splits are assigned to.
- **`context_valid_ratio` is measured over the largest context** an anchor
  allows; a shorter `L` chosen downstream may have a different ratio, which is
  why per-step validity stays in the grid.
