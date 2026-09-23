# Vocal action grid (v0)

## What it is

`conv-wm build vocal-action-grid` samples the continuous focal vocal state on a
regular control grid and labels each step with the vocal action the wearer's
behaviour logged during it:

```text
continuous control focal state  (SPEAKING / SILENT / UNKNOWN, exact seconds)
      -> regular Δ = 100 ms control grid
      -> logged vocal action proxy   (NO_EVENT / ONSET / OFFSET, or masked)
```

Its only semantic input is the versioned output of
`conv-wm build control-focal-voice-state`
([`control_focal_voice_state.md`](control_focal_voice_state.md)), which is the
native timeline ([`native_focal_voice_state.md`](native_focal_voice_state.md))
requantized at the same Δ by sub-step silence bridging. This builder never
merges anything itself: the only transform lives one layer up, and the report
records which one ran and what it changed.

It never reads Ego4D or EgoCom annotations, never opens media, and uses no
detector, speaker model or pseudo-label. The same algorithm runs on both
corpora; nothing branches on a dataset name.

## The grid

`decision_step_s = Δ = 0.100`, `t_k = k · Δ` on the recording's canonical
clock. Each slot is the half-open interval `[t_k, t_k + Δ)`: an event exactly
at `t_k` belongs to slot `k` with `tau_s = 0`, an event exactly at `t_k + Δ`
belongs to slot `k + 1`. Only slots entirely inside the timeline are emitted;
the partial tail is dropped and reported as `trailing_duration_dropped_s`
(`0 ≤ value < Δ` per recording).

Δ is a **control step**, not a video frame rate, not an audio sample period,
and not a prediction horizon `H`. Native timestamps are never rounded onto the
grid: `event_time_s` stays exactly as annotated and `tau_s = event_time_s − t_k`
records where inside the step the event happened.

Binary floating point makes `k · Δ` differ from the true grid point by an ULP
(`3 × 0.1 > 0.3`), so slot assignment and `tau_s` use a 1 ns tolerance
(`BOUNDARY_TOLERANCE_S`). Native annotations have millisecond resolution at
best, so this can never move a real event; it only prevents an event annotated
on a boundary from falling into the previous slot or producing a `tau` of −5e-17.

## Action semantics

`focal_state_before` is the state immediately before `t_k`.

| state before | resolved transitions in the slot | action |
| --- | --- | --- |
| `SILENT` | none | `NO_EVENT` |
| `SILENT` | exactly one `SILENT → SPEAKING` | `ONSET` |
| `SPEAKING` | none | `NO_EVENT` |
| `SPEAKING` | exactly one `SPEAKING → SILENT` | `OFFSET` |
| `UNKNOWN` | — | masked |

`NO_EVENT` means "hold the current vocal state for this step". It is neither
`SILENCE` nor `SPEAKING`: the state is carried by `focal_state_before`. The
action space is exactly `NO_EVENT`, `ONSET`, `OFFSET` — a slot the vocabulary
cannot express is **masked**, never given a fourth label.

Transitions involving `UNKNOWN` (`UNKNOWN → SPEAKING`, `SPEAKING → UNKNOWN`,
`UNKNOWN → SILENT`) are not actions. They only make slots invalid.

## Masking

| `mask_reason` | when |
| --- | --- |
| `recording_start` | `t_k` is the first instant of the timeline: no state before it |
| `unknown_state` | the state immediately before `t_k` is `UNKNOWN` |
| `unknown_within_slot` | a transition to or from `UNKNOWN` falls inside the slot |
| `compound_transition` | more than one resolved transition falls inside the slot |
| `invalid_timeline` | the recording's source timeline has a gap, an overlap or no interval |

A masked row keeps `action = null`, `action_valid = false` and its reason, so
the table stays auditable. `OFFSET + ONSET` in one slot is **never** silently
turned into `NO_EVENT`.

## Micro-gaps: measured here, bridged one layer up

The measurement that motivated the control layer is still produced, on both
timelines. Every `SPEAKING → SILENT → SPEAKING` gap shorter than Δ is
classified:

- **cross_slot** — the two events land in different slots, both representable;
- **same_slot** — they collide in one slot, which v0 masks as
  `compound_transition`.

Whether the *same physical pause* is one or the other depends on the grid's
phase, not on the pause: that arbitrariness is exactly what sub-step silence
bridging removes. On the control timeline, `sub_delta_gap_count` is therefore
**0** for both corpora, and the native counts survive in
`native_grid_comparison`.

`sub_delta_gap_analysis.parquet` keeps one row per remaining gap and
`compound_slots.parquet` one row per compound slot with its event times, types
and state `pattern`.

## Remaining compound slots

A slot holding more than one resolved transition is still masked, never
relabelled. After bridging, what remains is the compound the vocabulary
genuinely cannot express: a **short speech burst**,
`SILENT-SPEAKING-SILENT` inside one step. Those are deliberately kept
(see [`control_focal_voice_state.md`](control_focal_voice_state.md)), so the
report counts compound slots by pattern:

| pattern | meaning | after bridging |
| --- | --- | --- |
| `SPEAKING-SILENT-SPEAKING` | a sub-step silence | 0 by construction |
| `SILENT-SPEAKING-SILENT` | a short speech burst | kept and masked |
| longer chains | several events in one step | 0 by construction |

## Dataset differences are preserved

The two corpora reach this builder with different upstream conventions and the
builder does not harmonize them:

- **Ego4D** states come from native AV3 `voice_segments`, whose episodes can
  already absorb short internal pauses — so few sub-Δ gaps survive;
- **EgoCom** states come from speaker-attributed word intervals, which expose
  inter-word micro-gaps explicitly.

Any difference in compound-slot rates between the two is a property of the
annotations, not of the grid.

## Output schema

`${paths.processed}/vocal_action_grid/<dataset>/vocal_action_grid.parquet`

| column | meaning |
| --- | --- |
| `dataset`, `recording_id`, `sync_group_id`, `view_id`, `wearer_id` | identity, carried from the native state layer (ego-only: `actor_id = wearer_id`) |
| `decision_index`, `decision_time_s`, `slot_end_s` | `k`, `k · Δ`, `k · Δ + Δ` |
| `focal_state_before` | `SILENT`, `SPEAKING` or `UNKNOWN` immediately before `t_k` |
| `action` | `NO_EVENT`, `ONSET`, `OFFSET`, or null when masked |
| `action_valid` | false exactly when the slot is masked |
| `event_time_s`, `tau_s` | exact event time and its offset inside the step; null for `NO_EVENT` and masked slots |
| `mask_reason` | null exactly when `action_valid` |
| `source_control_state_schema_version`, `source_native_state_schema_version`, `action_schema_version` | the schemas this row was produced under |

No confidence, score, floor state, addressee or outcome column exists.

## Invariants (enforced at build time)

The build fails rather than writing a table that violates any of these:

- a valid row carries an action from the vocabulary and no mask reason; a
  masked row carries no action and a non-null reason;
- `NO_EVENT` has no `event_time_s` and no `tau_s`, and a known state before;
- `ONSET` comes from `SILENT`, `OFFSET` from `SPEAKING`, each with exactly one
  matching transition, `tau_s ∈ [0, Δ)` and `event_time_s = t_k + tau_s`;
- `(recording_id, decision_index)` is unique, `decision_index` increases within
  a recording, and `decision_time_s = decision_index · Δ`.

## What these labels are

Actions are **observational logged behaviour proxies**: what the wearer was
annotated to have done, sampled on a control grid. They are not randomized
causal interventions, and a model trained on them learns the logging policy's
statistics, not the effect of intervening.

**Observation must stop strictly before `t_k`.** The label of slot `k` is
defined on `[t_k, t_k + Δ)`; a dataloader that lets the model observe any part
of that window leaks the label. The action grid deliberately carries only
`decision_time_s` and `slot_end_s` so the boundary is explicit.

## Run and outputs

```bash
uv run conv-wm build native-focal-voice-state --dataset all    # prerequisite
uv run conv-wm build control-focal-voice-state --dataset all   # prerequisite
uv run conv-wm build vocal-action-grid --dataset all
uv run conv-wm build vocal-action-grid --dataset ego4d
```

The builder refuses to run if the control state artifact is missing, carries a
different `control_state_schema_version`, or no longer matches the checksum its
own report recorded — a stale layer is an error, never a silent input. The same
check is applied to the native timeline the control report points at, which the
build also grids as a diagnostic (`native_grid_comparison`); that grid is a
measurement and is never written as a table, so no extra CLI mode exists for
it.

`${paths.reports}/vocal_action_grid/<dataset>/`:

- `summary.parquet` — one row per recording: slot and action counts, mask
  counts by reason, compound and sub-Δ gap counts, dropped trailing time;
- `compound_slots.parquet` — every slot with more than one resolved
  transition, with its event times and types;
- `sub_delta_gap_analysis.parquet` — every native sub-Δ gap with its two slot
  indices and its cross/same-slot classification;
- `report.json` — the dataset statistics (action counts, `NO_EVENT` given
  each state, masks by reason, native vs represented vs masked transitions,
  compound-slot distribution, `tau` quantiles per action, recordings affected,
  sub-Δ gap analysis, compound slots by pattern) plus `native_grid_comparison`
  — the same statistics for the untransformed native timeline and the deltas —
  plus `control_transform` (which transform ran upstream and how much silence
  it bridged) and the lineage: `action_schema_version`,
  `source_control_state_schema_version`, `source_native_state_schema_version`,
  `decision_step_s`, `config_checksum`, `git_commit`, `git_dirty`, command,
  `created_at`, the input control timeline/report and native timeline
  checksums, the manifest and `uv.lock` checksums, the output checksums, the
  source annotation versions and the `lineage_chain`.

The build is deterministic and takes seconds (~2.7 M slots, ~12 MB of Parquet).

The grid is the last per-step layer: the model-ready stage
([`model_ready.md`](model_ready.md)) indexes the windows it supports without
copying any of it.

## Results (2026-09-23)

Every column is one grid of the same recordings: **native** is the diagnostic
grid of the untransformed native timeline, **control** is the production grid
written by this build. The slot counts are identical on both sides — bridging
removes interval boundaries, never time.

| | Ego4D native | Ego4D control | EgoCom native | EgoCom control |
| --- | --- | --- | --- | --- |
| slots | 1 316 810 | 1 316 810 | 1 387 790 | 1 387 790 |
| **valid action ratio** | 0.999651 | **0.999665** | 0.974407 | **0.997762** |
| `NO_EVENT` | 1 289 632 | 1 289 669 | 1 282 987 | 1 319 275 |
| `ONSET` / `OFFSET` | 13 359 / 13 359 | 13 350 / 13 350 | 34 443 / 34 842 | 32 696 / 32 713 |
| source transitions | 26 777 | 26 721 | 144 176 | 69 596 |
| represented / masked | 26 718 / 57 | 26 700 / 19 | 69 285 / 74 891 | 65 409 / 4 187 |
| **compound slots** | **20** | **1** | **34 495** | **2 083** |
| compound transitions | 40 | 2 | 74 869 | 4 166 |
| — `SPEAKING-SILENT-SPEAKING` | 19 | 0 | 28 246 | 0 |
| — `SILENT-SPEAKING-SILENT` | 1 | 1 | 2 335 | 2 083 |
| — longer chains | 0 | 0 | 3 914 | 0 |
| sub-Δ gaps (same / cross-slot) | 28 (19 / 9) | 0 | 37 290 (33 699 / 3 591) | 0 |
| recordings with a compound slot | 15 / 439 | 1 / 439 | 173 / 175 | 170 / 175 |

EgoCom's masked slots fall from 35 518 to 3 106 and its valid action ratio from
97.44 % to **99.78 %**; Ego4D, which had almost nothing to bridge, moves from
99.9651 % to 99.9665 %.

### Where EgoCom's compounds went

32 412 compound slots disappear. Of the 34 495 native ones, **32 160 contained
at least one sub-Δ silence** (28 246 plain `SPEAKING-SILENT-SPEAKING` plus
3 914 longer chains that include one) and every one of them is gone. The
remaining 252 were short bursts whose *surrounding* silences were themselves
sub-Δ, so the burst was absorbed into one continuous `SPEAKING`. The 33 699
same-slot micro-gaps therefore account for the drop almost exactly — the small
excess over 32 160 is slots that held more than one micro-gap.

What survives is what the vocabulary genuinely cannot express at Δ = 100 ms:
**2 083 short speech bursts** in EgoCom and **1** in Ego4D, all
`SILENT-SPEAKING-SILENT`. They are kept on purpose and stay masked.

No threshold was introduced after reading these numbers, and no rule other than
sub-step silence bridging was applied.
