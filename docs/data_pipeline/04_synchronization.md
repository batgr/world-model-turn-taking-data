# Technical A/V alignment

## Decision

Audio and video share a compatible container origin in the audited population.
All 369 files have an exact audio-minus-video start offset of 0 seconds, so no
global A/V offset correction is applied. Native audio and video PTS remain the
alignment clocks.

This is a technical container-timeline conclusion. It does not prove lip sync or
content-level audiovisual alignment.

## Measurements

`compute_av_sync_metadata` derives, from the media metadata table:

- `av_start_offset_sec = audio_start_time_sec - video_start_time_sec`
- `audio_end_time_sec = audio_start_time_sec + audio_duration_sec`
- `video_end_time_sec = video_start_time_sec + video_duration_sec`
- `av_end_delta_sec = audio_end_time_sec - video_end_time_sec`
- `abs_av_end_delta_sec`

| Dataset | Files | Zero start offsets | Minimum end delta (s) | Median (s) | Mean (s) | Maximum (s) | Maximum absolute (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego4D | 194 | 194 | -0.700667 | -0.001333 | 0.129698 | 0.589333 | 0.700667 |
| EgoCom | 175 | 175 | -0.306326 | -0.019342 | -0.040280 | 0.029342 | 0.306326 |

Audio/video duration or end differences are measurements, not automatic
synchronization failures. AAC priming and the PCM-vs-PTS effects documented in
[`03_temporal_audit.md`](03_temporal_audit.md) are among the reasons that
stream endpoints require careful interpretation. No evidence supports a
blanket correction. A duration mismatch is a container fact; a perceptual
lip-sync error would require content-level evidence (e.g. SyncNet), which is
not collected because nothing in the technical audit calls for it.

## Run and outputs

```bash
uv run conv-wm audit sync
```

Reports:

- `${paths.reports}/temporal/av_sync/av_sync_files.parquet`
- `${paths.reports}/temporal/av_sync/av_sync_summary.json`

## Deferred validation

- Cross-view synchronization is deferred until an architecture consumes
  multiple camera views simultaneously.
- Perceptual/content-level A/V synchronization is deferred.
- Fine A/V drift estimation or correction is deferred unless later evidence
  demonstrates a need.

Downstream multimodal alignment must use native timestamps and retain the local
audio discontinuities reported by the audio timeline audit.
