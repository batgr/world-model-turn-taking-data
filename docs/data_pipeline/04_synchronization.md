# Phase B4: technical A/V synchronization

## Decision

Audio and video share a compatible container origin in the audited population.
All 369 files have an exact audio-minus-video start offset of 0 seconds, so Phase B
applies no global A/V offset correction. Native audio and video PTS remain the
alignment clocks.

This is a technical container-timeline conclusion. It does not prove lip sync or
content-level audiovisual alignment.

## Measurements

`compute_av_sync_metadata` derives:

- `av_start_offset_sec = audio_start_time_sec - video_start_time_sec`
- `audio_end_time_sec = audio_start_time_sec + audio_duration_sec`
- `video_end_time_sec = video_start_time_sec + video_duration_sec`
- `av_end_delta_sec = audio_end_time_sec - video_end_time_sec`
- `abs_av_end_delta_sec`

The B4 runner verified the official B1 metadata report and completed in 0.90
seconds:

| Dataset | Files | Zero start offsets | Minimum end delta (s) | Median (s) | Mean (s) | Maximum (s) | Maximum absolute (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego4D | 194 | 194 | -0.700667 | -0.001333 | 0.129698 | 0.589333 | 0.700667 |
| EgoCom | 175 | 175 | -0.306326 | -0.019342 | -0.040280 | 0.029342 | 0.306326 |

Audio/video duration or end differences are measurements, not automatic
synchronization failures. AAC priming and the B3 PCM-vs-PTS effects are among the
reasons that stream endpoints require careful interpretation. No evidence in B4
supports a blanket correction.

## Run and outputs

From the repository root:

```bash
uv run python -m conv_wm.data.audits.run_av_sync
```

Reports:

- `${paths.reports}/temporal/av_sync/av_sync_files.parquet`
- `${paths.reports}/temporal/av_sync/av_sync_summary.json`

## Deferred validation

- B5 cross-view synchronization is deferred until an architecture consumes
  multiple camera views simultaneously.
- Perceptual/content-level A/V synchronization is deferred.
- Fine A/V drift estimation or correction is deferred unless later evidence
  demonstrates a need.

Downstream multimodal alignment must use native timestamps and retain the local
audio discontinuities reported by B3.
