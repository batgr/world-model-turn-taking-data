# Temporal and media audits

## Objective and scope

The media metadata, video timeline and audio timeline audits establish the
timestamp semantics needed for annotation integrity and multimodal sampling.
They measure media structure and record quality-control flags; they do not
alter raw media, resample streams or correct timestamps. The technical A/V
alignment audit is documented in [`04_synchronization.md`](04_synchronization.md).

The audits cover all 369 media files: 194 Ego4D and 175 EgoCom. Cross-view
synchronization is deferred because the initial architecture does not consume
multiple camera views simultaneously.

Run them with `conv-wm audit media`, `conv-wm audit video` and
`conv-wm audit audio` (implementation: `conv_wm.data.audits.media_metadata`,
`conv_wm.data.audits.video_timeline`, `conv_wm.data.audits.audio_timeline`;
measurements: `conv_wm.data.media`).

## Downstream temporal constraint

> Long Ego4D audio cannot in general be assigned absolute time by naïvely
> concatenating decoded PCM and using `sample_index / sample_rate`.

Native PTS is the only temporal authority. The audio timeline audit measures,
per file, how far concatenated PCM drifts from native PTS
(`final_cumulative_pcm_drift_ms`, up to 1.33 s in this corpus) and where the
largest transient offset occurs (`max_abs_cumulative_pcm_drift_time_sec`).
Any downstream audio loader must either follow PTS or apply the documented
per-file offsets; final model-time resampling is not part of this pipeline.

## Media metadata audit

`conv-wm audit media` probed all 369 files successfully. Every file has one video
stream and one AAC audio stream. Ego4D has 66 files at 32 kHz, 10 at 44.1 kHz,
and 118 at 48 kHz; all 175 EgoCom files use 44.1 kHz. Video is 30 FPS except for
two 60 FPS EgoCom files. Container, video, and audio start timestamps are exactly
zero in all files.

Audio-minus-video duration differences are retained as measurements rather than
treated as synchronization failures:

| Dataset | Files | Minimum (s) | Q1 (s) | Median (s) | Mean (s) | Q3 (s) | Maximum (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego4D | 194 | -0.700667 | -0.102667 | -0.001333 | 0.129698 | 0.437667 | 0.589333 |
| EgoCom | 175 | -0.306326 | -0.032336 | -0.019342 | -0.040280 | -0.003674 | 0.029342 |

Reports:

- `${paths.reports}/temporal/media_metadata/media_metadata.parquet`
- `${paths.reports}/temporal/media_metadata/summary.json`

## Video timeline audit

The population audit samples the start, middle and end of every stream plus
one 10-second window centred on each boundary the dataset declares (Ego4D:
every 300 s), so deterministic joins cannot escape a start/middle/end sample.
All 369 files and all 2,186 probed windows (1,083 of them boundary windows)
were compatible with constant-frame-rate timing; there were no sampled
suspects. The 300 s stitch grid that shapes the Ego4D audio timeline leaves
no trace on the video timeline. A targeted full-frame validation covered 18 files:
14 Ego4D files, including all 12 uncommon 544-pixel-wide files, and four EgoCom
files, including both 60 FPS files. All 18 full scans were also CFR-compatible.
That validation took 17 minutes 23 seconds.

This evidence permits FPS to describe cadence, but not to replace timestamps.
Native video PTS remains the temporal source of truth; downstream code must not
reconstruct video time as `frame_index / FPS`.

Reports:

- `${paths.reports}/temporal/video_timeline/sampled_timeline.parquet`
- `${paths.reports}/temporal/video_timeline/file_summary.parquet`
- `${paths.reports}/temporal/video_timeline/summary.json`

The targeted full-scan evidence is retained in
`notebooks/02_temporal_media_audit.ipynb` and was not repeated because the stable
source and tests revealed no video timeline defect.

## Audio timeline audit

### Measurement model

The authoritative population audit uses `ffprobe -show_packets` over each complete
file and keeps five concepts separate:

1. Packet-clock integrity compares `next_pts - current_pts` with the current
   packet duration.
2. Packet-duration variation records the observed duration distribution without
   declaring a gap.
3. PCM-vs-PTS offset compares each PTS increment, converted from integer PTS with
   an exact rational time base, with the independently validated decoded-frame
   sample count.
4. Codec boundary metadata records FFmpeg `skip_samples` and
   `discard_padding_samples` side data.
5. Dropout candidates and persistence. Every positive PCM-vs-PTS step strictly
   greater than the 1 ms drift threshold is a *dropout candidate*. The candidate
   is an `audio_dropout` only if the cumulative drift, measured relative to the
   pre-step baseline, never falls below `persistence_fraction` (0.5) of the step
   excess during the following `compensation_window_sec` (2.0 s, converted to
   packets with the decoded-frame reference). A candidate compensated inside
   that window is a *timestamp cadence*, not missing audio: its episode is
   reported as `compensated_timestamp_cadence`. An abnormally long packet
   duration (above the robust per-file fence) is retained as an observation on
   every event and in the file table, but it never decides the category. Both
   rules are generic: they use no dataset name and no sample rate.

Every threshold comparison in this audit is strict (`value > threshold`); a
value equal to a threshold is not counted above it. The 1 ms threshold is an
explicit QC interpretation threshold. Raw step errors and packet-duration
measurements remain available below it. Cumulative drift is a file-level metric:
the file report stores `final_cumulative_pcm_drift_ms`,
`max_abs_cumulative_pcm_drift_ms`, and the native timestamp of that maximum in
`max_abs_cumulative_pcm_drift_time_sec`; the event report contains only local
episode measurements plus, for each dropout, the minimum residual inside its
compensation window (`dropout_min_residual_in_window_samples`).

### Population results

The four-worker full scan processed 20,779,635 packets in 2,411.48 seconds
(40 minutes 11.48 seconds). All 369 files probed successfully. Timestamp and
duration coverage are 100%, all PTS sequences are strictly monotonic, and the
packet clock has zero gaps, zero overlaps, and a maximum absolute packet-timeline
error of zero samples.

Targeted `-show_frames` validation covered 57 scientifically selected windows. It
includes start and end boundaries for every dataset/sample-rate regime,
representative anomaly families, the largest inferred offsets, one dropout per
affected dataset/rate, the largest compensated-cadence step, and four windows in
each of the ten Ego4D 44.1 kHz files. Every decoded frame contained 1,024
samples; all 57 modal references matched, and no 960-sample AAC frame was
observed. The population
contains AAC only, so the packet audit uses 1,024 samples with
`targeted_show_frames_mode` provenance. This conclusion applies to the audited
population and must be revalidated if another codec or AAC regime is added.

| Dataset/rate | Files | Any PCM-step variation | Bounded below 1 ms | Significant transient offset | Significant final offset | Stitch files | Early-boundary files | Dropout files | Maximum transient offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego4D 32 kHz | 66 | 65 | 0 | 65 | 65 | 55 | 0 | 10 | 58,624 samples / 1,832.000 ms |
| Ego4D 44.1 kHz | 10 | 10 | 0 | 10 | 9 | 0 | 0 | 7 | 6,205 samples / 140.703 ms |
| Ego4D 48 kHz | 118 | 63 | 5 | 58 | 58 | 0 | 17 | 8 | 4,800 samples / 100.000 ms |
| EgoCom 44.1 kHz | 175 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 samples / 0 ms |

Across Ego4D, 138 files have at least one non-1,024-sample PTS increment, 133
cross the 1 ms transient threshold, and 132 retain a final offset above 1 ms.
Those flags are orthogonal: “variation” is an observed measurement and does not
by itself identify a packet loss or a single cause.

Final drift is summarized by dataset and sample rate below. Quantiles are ordered
`min/p05/p25/p50/p75/p95/max`; threshold counts use absolute drift, and the
half-frame threshold is calculated separately from each file's video FPS.

| Dataset/rate | Signed final drift quantiles (ms) | Absolute drift p50/p95/max (ms) | >1 ms | >half frame | >100 ms | >500 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ego4D 32 kHz | -192 / -192 / -192 / -112 / -64 / 854 / 1,330.656 | 160 / 854 / 1,330.656 | 65 | 65 | 42 | 7 |
| Ego4D 44.1 kHz | -131.859 / -128.227 / -117.262 / -45.011 / -0.181 / 27.714 / 41.633 | 64.705 / 128.227 / 131.859 | 9 | 6 | 4 | 0 |
| Ego4D 48 kHz | -97.333 / -63.467 / -19.266 / 0 / 0 / 5.445 / 100 | 0.625 / 69 / 100 | 58 | 36 | 0 | 0 |
| EgoCom 44.1 kHz | 0 / 0 / 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 | 0 | 0 |

The JSON report retains the complete signed and absolute seven-quantile
distributions. Threshold counts are strict: the one Ego4D 48 kHz file whose final
drift is exactly 4,800 samples (100.000 ms) is counted above 1 ms and above half
a frame, but not above 100 ms.

The event table contains 74 `audio_dropout` events, each a single persistent
positive step. Dropout duration is that step's excess over the decoded-frame
reference. Candidate accounting separates persistence from packet length:

| Dataset/rate | Files with dropouts | Dropout events | Candidate steps | Compensated candidates | Abnormally long packets (observation) | Dropout duration quantiles (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ego4D 32 kHz | 10 | 40 | 40 | 0 | 38 | 1.313 / 22.866 / 57.344 / 90.656 / 489.672 / 634.658 / 952 |
| Ego4D 44.1 kHz | 7 | 25 | 108,106 | 108,081 | 95,962 | 1.270 / 2.172 / 10.159 / 16.395 / 18.118 / 21.995 / 22.630 |
| Ego4D 48 kHz | 8 | 9 | 26 | 0 | 12 | 33.333 / 34.933 / 37.333 / 37.333 / 62.667 / 93.279 / 99.021 |
| EgoCom 44.1 kHz | 0 | 0 | 0 | 0 | 0 | n/a |

At 48 kHz, 26 persistent steps yield 9 dropouts because the other 17 sit 44--54 ms
after the stream origin inside the AAC priming context and carry the
codec-boundary label by precedence (three of them are also abnormally long
packets). At 44.1 kHz, 99.98% of the 108,106 candidate steps are compensated
inside the two-second window; the 25 that persist are 1.3--22.6 ms and come from
all seven files, including one to three per dense-cadence file.

The 44.1 kHz packet pattern has two systematic regimes. Four files have a robust
per-file nominal PTS duration of 885 samples (MAD 10--11); 15.33--15.51% of their
packet durations are abnormally long. Those long packets are the compensating
half of a short/long cadence: roughly nine 885-sample increments are followed by
one increment near 2,300 samples, the cumulative PCM-vs-PTS error oscillates
within about 30 ms and returns to baseline within a fraction of a second. The
refined audit reports this as 16 `compensated_timestamp_cadence` episodes
(median 23,728 steps, median net drift -72 samples), not as dropouts. Six files
have a robust nominal duration of 1,024 samples with sparse deviations; three
have no abnormally long durations and three have one to eight. Signed final
drifts across the ten files range from -131.859 to +41.633 ms, so neither a
single offset nor a simple rate correction describes the group. The forty
targeted windows yielded decoded frames of exactly 1,024 samples throughout, as
did the window centred on the largest compensated-cadence step. This supports
the measured PCM-vs-PTS discrepancy but does not support inventing a
correction; native PTS remains the clock.

Codec side data is present in every file. All 194 Ego4D files report 2,048
`skip_samples`. Of the 175 EgoCom files, 174 report 1,024 and one reports 142.
All 369 files report zero discard-padding samples.

### Supported anomaly taxonomy

- **Internally coherent packet clock.** Packet PTS and packet duration agree for
  every measured transition. The older full-packet result of “zero drift” was a
  packet-clock result, not a decoded-PCM result.
- **Small bounded variation.** Five Ego4D 48 kHz files stay below 1 ms. Their
  maximum cumulative offsets range from 5 to 41 samples (at most 0.854 ms), even
  though one file has many compensating small steps. This is observable but is
  harmless for the current turn-taking prototype when native PTS is preserved.
- **Audio dropout.** Seventy-four single-step events in 25 Ego4D files: a
  positive PCM-vs-PTS step above 1 ms whose excess persists (at or above half the
  step) through the following two seconds. The affected decoded validation windows
  remain 1,024-sample AAC frames, so the decoded stream is contiguous while the
  PTS clock jumps forward; concatenated PCM falls behind native PTS by the step
  excess. This is a persistent clock discontinuity, not direct evidence of
  audible silence. Each event records its minimum residual inside the window and
  whether the window was truncated by the end of the stream.
- **Compensated timestamp cadence.** Sixteen episodes in four Ego4D 44.1 kHz
  files where positive candidate steps are cancelled inside the compensation
  window by surrounding short steps. Abnormally long packet durations
  (`max(median + max(6 * MAD, 0.5 * median), decoded reference + 1 ms)` fence)
  are retained as an observation on every event and never decide the category:
  95,962 of the 95,979 long packets observed at 44.1 kHz belong to this cadence.
- **Deterministic 32 kHz stitch discontinuity.** The full scan found 225 events in
  55 Ego4D 32 kHz files, each with a 1,023-sample maximum step and -1,024-sample
  net PCM-clock change. They lie within one 32 kHz tick of the empirically
  supported 300-second Ego4D stitch grid. Concatenated PCM can accumulate about
  32 ms per event.
- **Early codec-boundary event.** Seventeen persistent positive steps in Ego4D
  48 kHz files sit 44--54 ms after the stream origin, in the AAC priming context
  (`skip_samples` = 2,048). They satisfy the dropout persistence rule, but the
  codec-boundary label takes precedence because the cause is known; magnitudes
  are 1.06--11.12 ms.
- **Other PCM timestamp variation.** Ninety-two significant episodes do not
  meet the dropout, cadence, stitch-grid, or early codec-boundary definitions.
  They remain explicit measurements and inputs to the temporal representation
  rather than being silently corrected or excluded here.

The authoritative event table therefore has 424 rows: 225 stitch
discontinuities, 74 audio dropouts, 16 compensated-cadence episodes, 17 early
codec-boundary events, and 92 other PCM timestamp variations. It contains no
cumulative-offset peak rows; cumulative drift magnitude and its native timestamp
(`max_abs_cumulative_pcm_drift_time_sec`) are file-level fields.

The earlier decoded start/middle/end audit remains provenance, not the
authority. Its 143 Ego4D `gap_or_overlap` windows mixed bounded jitter, priming,
stitch events, and other variations; a sampled `continuous` verdict could also
miss fixed boundaries. The full packet audit plus targeted decoding supersedes
that binary interpretation.

Reports:

- `${paths.reports}/temporal/audio_timeline/audio_packet_timeline_files.parquet`
- `${paths.reports}/temporal/audio_timeline/audio_packet_timeline_events.parquet`
- `${paths.reports}/temporal/audio_timeline/audio_packet_timeline_summary.json`
- `${paths.reports}/temporal/audio_timeline/audio_decode_validation.parquet`
- `${paths.reports}/temporal/audio_timeline/audio_decode_validation_summary.json`

Event rows carry the interpretation context in generic columns:
`nearest_known_boundary_index`, `nearest_known_boundary_time_sec`,
`distance_to_known_boundary_sec`, `known_boundary_tolerance_sec`,
`near_known_boundary`, `early_boundary_tolerance_sec`,
`has_early_boundary_context`. The boundary grid itself is declared by the
dataset (`DatasetSpec.audio.known_boundary_grid`) and recorded under
`parameters.datasets` in the summary.

### Unresolved behaviour

- The physical cause of the dense 44.1 kHz cadence and of the sparse −5-sample
  steps is unknown; both are measured, neither is corrected.
- Whether the 74 persistent dropouts are audible silence, duplicated content or
  pure timestamp jumps cannot be decided from packets; decoded frames stay
  nominal, so the discrepancy is between clocks, not inside the PCM.
- AAC priming is handled by FFmpeg 7.1; the reported `skip_samples`, the
  17 early codec-boundary events and `-read_intervals` behaviour must be
  re-validated when FFmpeg changes (see `provenance.ffprobe_version`).

The older sampled outputs in the same directory are retained for provenance.

## Run

From the repository root:

```bash
uv run python -m conv_wm.data.audits.run_media_metadata
uv run python -m conv_wm.data.audits.run_video_timeline
uv run python -m conv_wm.data.audits.run_audio_timeline
```

The audio runner performs complete packet scans and a small deterministic targeted
decode. Four workers were retained after an eight-file benchmark improved from
69.29 seconds sequentially to about 28 seconds.

## Temporal contract

The following rules are binding inputs to later capabilities (temporal
representation, model-ready packaging):

- Raw media is immutable; the temporal audits report measurements and never
  rewrite media.
- Native video PTS and native audio PTS are the temporal sources of truth.
- Annotation integrity validates annotation timestamps against these media
  timelines (see [`05_annotation_audit.md`](05_annotation_audit.md)).
- Never reconstruct video time solely as `frame_index / FPS`.
- Long-file Ego4D PCM must not be aligned using naive
  `sample_index / sample_rate`; the temporal representation must preserve or
  reconstruct the mapping from decoded samples to native audio PTS.
- Preserve discontinuities and expose masks or timestamp-aware segment boundaries
  to downstream consumers. Do not interpolate across them automatically.
- Do not apply a global A/V offset or fine drift correction without supporting
  evidence.
- Resampling, if required by a model, happens in the temporal representation and
  must preserve the mapping to native PTS.
- Cross-view synchronization and perceptual/content-level A/V synchronization are
  deferred.

No file is excluded: all probes succeeded and the anomalies are representable
with timestamps and masks. The unresolved risk is downstream code that
concatenates decoded PCM and assigns time from sample index alone; the temporal
representation must consume the audio timeline file/event reports or derive
equivalent timestamp-aware segments.
