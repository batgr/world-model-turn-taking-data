# Glossary

Terms used consistently across code, report columns and documentation. When a
report column and a term differ, the column name is given in code style.

## Media and timelines

- **Packet** — one compressed unit of a stream as stored in the container, read
  with `ffprobe -show_packets` without decoding. Carries a **PTS** and a
  **packet duration** (`duration`, in time-base ticks).
- **PTS** (presentation timestamp) — the native time of a packet or frame in
  the stream's **time base** (a rational number of seconds per tick, e.g.
  `1/48000`). PTS is the temporal source of truth of the pipeline; nothing
  downstream may reconstruct time from frame or sample indices.
- **Decoded frame** — the PCM block a decoder emits for one audio packet
  (`nb_samples` samples; 1024 for AAC-LC). Measured with `ffprobe -show_frames`
  on short windows only, because decoding is slow.
- **Sample** — one PCM sample; the unit in which audio clock errors are
  reported (`*_samples`), alongside milliseconds (`*_ms`).
- **PTS step** — `next_pts − pts` between two adjacent packets that both carry
  a PTS. Steps are only formed between adjacent valid packets; missing fields
  reduce coverage instead of bridging gaps.
- **Packet clock** — the container's own timeline: `next_pts − pts − duration`
  per packet. Zero everywhere means the container is internally coherent
  (`packet_timeline_gap_count`, `packet_timeline_overlap_count`).
- **PCM clock** — the timeline a naïve consumer builds by concatenating decoded
  frames: `next_pts − pts − reference_decoded_samples` per step (the **PCM step
  error**).
- **Reference decoded samples** — the decoded frame size assumed by the PCM
  clock (`reference_decoded_samples`), validated by decoding
  (`reference_decoded_samples_source`).
- **Cumulative PCM drift** — running sum of PCM step errors. A *file property*:
  `final_cumulative_pcm_drift_ms` is the offset concatenated PCM carries at the
  end of the file; `max_abs_cumulative_pcm_drift_ms` its largest transient
  magnitude, located at `max_abs_cumulative_pcm_drift_time_sec` (native PTS
  seconds).
- **Timestamp jitter** — non-zero PCM step errors. *Small bounded* jitter stays
  within the drift threshold both per step and cumulatively
  (`has_small_bounded_timestamp_jitter`).
- **Timestamp cadence / variation** — PCM step errors above the threshold that
  do not persist. A **compensated timestamp cadence** episode contains
  positive candidate steps cancelled by surrounding short steps inside the
  compensation window (`compensated_timestamp_cadence`); other above-threshold
  episodes are `pcm_timestamp_variation`.
- **Abnormally long packet** — packet duration above the per-file fence
  `max(median + max(6·MAD, 0.5·median), reference + threshold)`. An
  observation only (`n_abnormally_long_packet_durations`,
  `n_abnormally_long_packets`); it never decides a category.
- **Dropout candidate** — a positive PCM step error strictly greater than the
  drift threshold (`n_dropout_candidate_steps`).
- **Audio dropout** — a candidate whose excess persists: the cumulative drift
  relative to the pre-step baseline never falls below `persistence_fraction`
  of the step inside `compensation_window_sec` (`audio_dropout`,
  `n_persistent_dropout_steps`). Concatenated PCM falls behind native PTS by
  the excess. Not proof of audible silence.
- **Stitch-related PCM discontinuity** — a non-dropout episode within
  `known_boundary_tolerance_sec` of a boundary declared by the dataset
  (`stitch_related_pcm_discontinuity`, `near_known_boundary`). The grid is
  dataset knowledge (Ego4D: every 300 s), not generic logic.
- **Codec boundary / priming** — AAC encoder delay. FFmpeg reports it as
  `skip_samples` side data (`discard_padding_samples` at the end). PCM events
  inside the priming context are `codec_boundary_pcm_event` regardless of
  persistence.
- **Episode** — consecutive non-zero PCM steps grouped into one event row
  (`pcm_step_episode`, `n_steps`). A persistent candidate becomes its own
  single-step `audio_dropout` row and splits the episode around it.
- **Event** — one row of the events table: a packet-clock error, an episode or
  a dropout. `event_type` is structural; `event_category` is the
  interpretation.
- **Timeline status** — whether a file yielded a measurement
  (`timeline_status`: `measured`, `insufficient_data`, `probe_error`).
- **Probe error** — ffprobe could not read the file; the record keeps
  `probe_ok = False` and the message in `probe_error` and the population audit
  continues.
- **Threshold comparison** — every threshold in the temporal audits is strict:
  `value > threshold` (`threshold_comparison = "strictly_greater"`). A value
  equal to a threshold is not counted above it.
- **Known boundary grid** — periodic positions a dataset declares as recording
  joins (`KnownBoundaryGrid`). The audio audit relabels events near them; the
  video audit samples one extra window on each.
- **CFR-consistent** — every video PTS step matches `1/fps` within one
  time-base tick (`cfr_consistent`); otherwise `vfr_or_irregular`.
- **A/V start offset** — `audio_start_time_sec − video_start_time_sec` from
  container metadata (`av_start_offset_sec`). A technical container fact, not
  perceptual lip-sync.

## Annotations

- **Annotation source** — one table of a dataset described by an
  `AnnotationSourceSpec`: scope, temporal coordinates, references, values,
  provenance, known limitations.
- **Scope** — what a row describes: `point_event`, `temporal_interval`,
  `clip`, `video`, `interaction`, `participant`, `participant_interaction`,
  `sequence_global`.
- **Temporal origin** — the zero of a source's timestamps: `media_start`,
  `clip_start`, `interaction_start`, `absolute`.
- **Entity reference** — columns that must exist in another source
  (`media`, `clip`, `interaction`, `participant`). Declared **unknown values**
  (e.g. `-1`) mean "unknown", not dangling.
- **Duration bounds** — the referenced duration a source's timestamps are
  compared with (`end_beyond_reference_duration`).
- **Anomaly** — one violated or informational constraint with a
  **severity**: `error` blocks downstream use, `warning` is a documented
  caveat, `info` records an expected property.
- **Downstream suitability** — `usable`, `usable_with_caveats` or `blocked`,
  derived from anomalies and known limitations.
- **Provenance (annotation)** — how values came to exist: `human_observed`,
  `deterministic_derived`, `transferred`, `pseudo_label`, `cluster_derived`,
  `model_inferred`. Carried forward, never inferred by the audit.
- **Cross-source coverage** — fraction of one source's intervals overlapping
  at least one interval of another source with the same entity key. Evidence
  of agreement, not identity.

## Native focal voice state

- **Focal voice state** — the camera wearer's vocal state on the canonical
  timeline: `SPEAKING`, `SILENT` or `UNKNOWN`. Derived from native
  annotations only; `actor_id = wearer_id` (ego-only prototype).
- **Source kind** — the native evidence behind an interval:
  `ego4d_voice_segments` (a vocal episode, possibly absorbing short pauses),
  `egocom_transcript` (a speaker-attributed word interval),
  `invalid_or_missing` (declared `UNKNOWN` region).
- **Canonical window** — the annotated time range of one recording on its
  dataset's canonical timeline; the timeline covers it exactly.
- **Native-annotation-derived state** — what v0 produces, as opposed to an
  exhaustive physical vocal-activity ground truth; absence of annotation is
  `SILENT` by contract, with the documented coverage limitations.

## Control focal voice state

- **Control focal voice state** — the native timeline requantized at Δ: what a
  controller running at `decision_step_s` can represent. Written as a separate
  artifact; the native one is immutable.
- **Sub-step silence bridging** — the one control transform:
  `SPEAKING → SILENT (g < Δ) → SPEAKING` becomes continuous `SPEAKING`. The
  condition is strict, so a gap of exactly Δ is kept. A quantization tied to the
  controller's resolution, **not** a correction of the annotation.
- **Bridged gap** — one native sub-Δ silence absorbed by that rule; each keeps a
  row in `bridged_gaps.parquet` and is counted on the interval that absorbed it
  (`bridged_gap_count`, `bridged_gap_total_duration_s`).
- **Short speech burst** — a `SILENT → SPEAKING → SILENT` vocalization; never
  filtered, whatever its duration. The symmetric rule is deliberately not
  applied.

## Vocal action grid

- **Decision step (Δ)** — the control period of the action grid, 0.100 s. Not
  a video frame rate, an audio sample period or a prediction horizon.
- **Slot** — one half-open control step ``[t_k, t_k + Δ)`` with ``t_k = k · Δ``;
  only slots entirely inside the timeline are emitted.
- **Action** — `NO_EVENT` (hold the current state), `ONSET`
  (`SILENT → SPEAKING`), `OFFSET` (`SPEAKING → SILENT`). The complete v0 space.
- **Masked slot** — a slot the vocabulary cannot express: `action` is null,
  `action_valid` false and `mask_reason` says why. Never a fourth action.
- **tau** — `event_time_s - t_k`, the position of the event inside its step;
  native timestamps stay exact, nothing is rounded onto the grid.
- **Compound slot** — a slot holding more than one resolved transition; masked
  in v0. Its **pattern** is the state chain it walks through
  (`SILENT-SPEAKING-SILENT` is a short speech burst,
  `SPEAKING-SILENT-SPEAKING` a sub-step silence, which bridging removes).
- **Sub-Δ gap** — a `SPEAKING → SILENT → SPEAKING` gap shorter than Δ,
  classified **cross_slot** (both events representable) or **same_slot**
  (they collide in one slot). Which one it is depends on the grid's phase, not
  on the pause: the control layer removes that arbitrariness upstream.
- **Native grid comparison** — the same statistics computed on the
  untransformed native timeline, reported next to the production ones so the
  effect of bridging is explicit. A diagnostic; never written as a table.
- **Logged behaviour proxy** — what an action label is: observed annotated
  behaviour sampled on a grid, not a randomized causal intervention.

## Model ready

- **Anchor** — a grid slot that can carry a training window: enough context
  before it, a complete future after it, inside one segment.
- **Segment** — a maximal run of consecutive `decision_index` values of one
  recording; windows never cross one.
- **max_context_steps** — the largest context an anchor supports,
  `min(available past+current steps, 50)`. The context length itself is chosen
  downstream, never materialized here.
- **Sample class** — `event` when the future horizon logs an action other than
  `NO_EVENT`, `background` otherwise. Metadata for downstream sampling; the
  natural distribution is preserved.
- **is_trainable** — both measured validity ratios satisfy the thresholds.
  Rejected anchors stay in the index with their ratios.
- **Split** — `train` / `validation` / `test`, assigned to whole conversation
  groups, preferring the release's own assignment; a deterministic seeded
  fallback covers sessions it does not.
- **Split leakage** — a conversation group whose sessions do not share one
  split. Reported, never silently repaired.

## Label sidecars

- **Label** — one entry of the registry (`family.label`): meaning, level,
  modalities, time reference, source kind, role, validity and storage.
  [`labels_registry.md`](labels_registry.md) lists them all.
- **Source kind** (label) — `native_annotation`, `deterministic`,
  `external_model` (never ground truth) or `human_annotation` (not available).
- **Fact** — what a dataset adapter declares it can provide (`speech`,
  `words`, `transcript`, `social.looking`, ...); a label is supported by a
  dataset exactly when the dataset provides all the facts it requires.
- **Extractor** — the unit of building (`speech`, `social`, `text`, `audio`,
  `video`): one directory of tables plus a deterministic `manifest.json`.
- **Participant index** — position in a recording's `participant_ids`; 0 is
  always the wearer.
- **Subframe** — one of the `S` equal parts of a decision cell (default 3,
  i.e. 30 Hz).
- **Reference instant** — the cell end `r_k = t_k + Δ` at which timing, next
  speaker and future labels are evaluated.
- **Last unique speaker / floor** — the participant who most recently spoke
  alone; a **floor change** is a change of it between two known participants,
  its **FTO** the start of the new holder's run minus the end of the previous
  holder's.
- **Censored** — a value that would need time beyond the observed span (the
  recording end or UNKNOWN): null and invalid, never zero.
- **Ego-solo mask** — time where the wearer is the only known speaker; the
  only time the wearer's prosody is measured on.

## Reports

- **Summary** — the JSON document an audit writes; always contains
  `summary_schema_version`, `provenance` and, when relevant, `parameters`.
- **Provenance (report)** — generation time (UTC), package version, git commit
  and dirty flag, Python, ffmpeg and ffprobe versions.
