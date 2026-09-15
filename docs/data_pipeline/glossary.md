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

## Reports

- **Summary** — the JSON document an audit writes; always contains
  `summary_schema_version`, `provenance` and, when relevant, `parameters`.
- **Provenance (report)** — generation time (UTC), package version, git commit
  and dirty flag, Python, ffmpeg and ffprobe versions.
