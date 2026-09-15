# Annotation integrity

## Objective

Determine whether each annotation source is semantically, temporally,
relationally and referentially trustworthy enough for later derivation and
analysis. The audit does not derive labels, does not decide training targets
and does not correct anything: it detects, classifies and documents.

Run with `uv run conv-wm audit annotations` (implementation:
`conv_wm.data.audits.annotations`; checks: `conv_wm.data.annotations`;
declarations: `conv_wm.data.datasets.<name>.ANNOTATIONS`). The media metadata
table is a prerequisite because bounds use measured stream durations.

## Architecture

Every dataset declares its sources with typed `AnnotationSourceSpec` objects:

| Declaration | Meaning |
| --- | --- |
| `scope` | what one row describes: point event, temporal interval, clip, video, interaction, participant, participant × interaction, sequence/global |
| `temporal` | start/end columns, unit (seconds, milliseconds, frames, samples), origin (media, clip, interaction, absolute), nullability, tolerance |
| `media_reference` | columns naming a media file, and how a media path maps to that key |
| `entity_references` | columns that must exist in another source (clip, interaction, participant), with declared "unknown" sentinels such as `-1` |
| `bounds` | which referenced duration the timestamps are compared with (a table's duration or start/end columns, or the measured media duration) |
| `identity` | columns that identify a row (duplicates are errors) |
| `value_fields` | payload columns preserved with their native meaning |
| `provenance`, `confidence_field` | how the values came to exist and, when present, a per-row confidence |
| `known_limitations` | facts about the source that no check can establish |

Generic checks then run without knowing the dataset: finite timestamps,
`start <= end`, duration distribution, negative starts, bounds against the
referenced duration, duplicate identities, dangling media and entity
references, and — for declared `CrossSourceComparison`s — directional
interval-overlap coverage per shared entity key. Each finding is an
`Anomaly` with a severity fixed by the declaration: `error` blocks downstream
use, `warning` is a documented caveat, `info` records an expected property.
A negative timestamp is never automatically invalid.

Outputs below `${paths.reports}/annotations/`:

- `annotation_integrity.json` — the contract: per dataset and source, its
  identity, scope, temporal coordinate system, media and entity relations,
  validated constraints, anomalies with counts, known limitations, validity and
  `downstream_suitability` (`usable`, `usable_with_caveats`, `blocked`);
- `annotation_sources.parquet`, `annotation_anomalies.parquet`,
  `annotation_cross_source.parquet` — the same as flat tables.

Sources are not merged into one universal table; each keeps its own columns
and semantics.

## Findings on the audited corpus

All 10 declared sources pass (no error-severity anomaly); every one is
`usable_with_caveats` except `tracking_paths`, which is `usable`.

### Ego4D (8 sources, clip timeline)

| Source | Scope | Rows | Caveats found |
| --- | --- | ---: | --- |
| `clips` | clip | 439 | 10 clips start at `video_start_sec = -0.0013 s` (one 768-tick rounding); clip durations 298.6–300.1 s; every `video_uid` resolves to a media file |
| `persons` | participant | 3,231 | camera wearer (person 0) has no face track and no looking annotation |
| `voice_segments` | interval | 40,383 | 260 intervals end up to 1.025 s beyond the clip duration (median overshoot 30 ms); 5 zero-duration intervals |
| `transcriptions` | interval | 39,191 | 351 intervals overshoot (max 1.025 s); 2,630 rows carry the unknown speaker `-1`; some clips have no transcription |
| `social_segments_talking` | interval | 12,453 | 97 intervals overshoot (max 1.025 s); 22 rows with speaker `-1`; `target` semantics unconstrained |
| `social_segments_looking` | interval | 0 | empty in the interim snapshot; gaze labels are shipped as separate per-frame files |
| `tracking_paths` | participant | 38,242 | none |
| `tracks` | point event (frames) | 5,523,640 | 490 boxes in 14 clips lie up to 40 frames beyond the clip's last frame |

No dangling clip or participant reference anywhere in Ego4D.

Cross-source agreement (interval overlap per `clip_uid` × person):

| Comparison | Left covered | Right covered | Shared keys | Left-only | Right-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| `voice_segments` vs `transcriptions` | 86.7 % (5,381 uncovered) | 89.3 % (4,183 uncovered) | 1,988 | 83 | 278 |
| `voice_segments` vs `social_segments_talking` | 30.6 % (28,011 uncovered) | 99.6 % (52 uncovered) | 1,122 | 949 | 11 |

Voice activity and transcription are independent annotations with asymmetric
coverage in both directions; they must be joined by overlap and person, never
by equal timestamps. Talking segments (who talks to whom) cover only 30.6 % of
voice activity while 99.6 % of them fall inside voice activity: addressee
labels exist for a subset of speech, and the 52 talking segments outside any
voice segment are unresolved.

### EgoCom (2 sources, conversation-part timeline)

| Source | Scope | Rows | Caveats found |
| --- | --- | ---: | --- |
| `video_info` | participant × interaction | 175 | `duration_seconds` is a declared integer, not the measured stream length; every `video_name` resolves to a media file |
| `ground_truth_transcriptions` | interval (word) | 359,536 | 177,573 timed rows (49.4 %; punctuation and empty tokens are untimed, declared nullable); no interval ends beyond the declared part duration (last word ends 0.2–3.7 s before it); 19,627 zero-duration words; **13,512 words in 13 conversation parts (6 physical conversations) are attributed to speaker 3 although `num_speakers = 2` and no recording exists for that speaker** |

The speaker-3 case is the one unresolved semantic anomaly of the corpus. It is
recorded as a `warning` (participant reference to `video_info`, the only
participant table EgoCom ships) with a known limitation; whether speaker 3 is
an unrecorded third participant or a non-participant voice must be decided
downstream, not by the audit.

## Unresolved semantics

- Ego4D overshoots of up to 1.025 s beyond a 300 s clip: annotation bounds
  extend slightly past the clip cut; consumers must clip or keep them
  explicitly.
- Ego4D `social_segments_talking.target` and EgoCom speaker 3: identities not
  fully specified by the release.
- Ego4D looking annotations: the interim table is empty; the per-frame label
  files are not yet part of the pipeline.
- Cross-source disagreement (uncovered voice segments, talking segments outside
  voice activity) is reported, not adjudicated.

## Provenance for later work

Every source carries `provenance = human_observed` except `clips`
(`deterministic_derived` from the benchmark release). No source has a
confidence field. Derived, transferred, pseudo-labelled, clustered or
model-inferred annotations must be added as new sources with their own
provenance and confidence field, as the synthetic `moodlab` dataset in the
tests demonstrates with `model_inferred` affect intervals.
