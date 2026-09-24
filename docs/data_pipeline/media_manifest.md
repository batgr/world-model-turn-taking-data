# Media manifest (v0)

## What it is

`conv-wm build media-manifest` writes one canonical table per dataset that
says where the raw media of each canonical recording lives:

```text
(dataset, recording_id)  ->  video_path, audio_path (relative to the corpus root)
```

A modelling repository joins it with the action grid on `recording_id` and
needs no knowledge of any corpus layout, native filename or clip convention.
All of that stays in the dataset adapters of this repository.

```text
vocal action grid ─┐  canonical recording ids
media metadata  ───┼─> <dataset>_media adapter -> media_manifest.parquet
adapter tables  ───┘  (Ego4D: clips_clean)
```

It runs after the action grid and before model-ready in `build all`.

## Schema (`media_schema_version = 1`)

| Column | Meaning |
| --- | --- |
| `dataset` | corpus key |
| `recording_id` | canonical recording id, identical to `action_grid.recording_id` |
| `video_path` | video file relative to the corpus root, or null |
| `audio_path` | *separate* audio file relative to the corpus root, or null |
| `media_offset_s` | `media_time_s = decision_time_s + media_offset_s` |
| `video_has_audio` | whether the video container has an audio stream (null if unprobed) |

One row per recording: a recording with both a video and an audio file is one
row, not two. `(dataset, recording_id)` is unique; several recordings may point
at the same file (Ego4D clips cut from one video), each with its own offset.

`audio_path` is null when the corpus ships no separate audio file. With
`video_has_audio = true` the consumer decodes the audio track of the video
container; no WAV file is ever generated to fill the column.

No other metadata is copied: frame rate, sample rate and duration are one
`ffprobe` away and are recorded, per file, by `conv-wm audit media`.

## Paths

Paths are POSIX, relative to the dataset's **corpus root**: its directory below
the raw root (`raw/EgoCom`, `raw/Ego4D`), recorded as `corpus_directory` in the
report and the dataset card. No absolute path, user name or mount point is
stored; the runtime joins its own configured root:

```text
<configured corpus root> / video_path
```

## Adapters

A dataset opts in with `DatasetSpec.media_records`, a callable
`(cfg, media_metadata, recording_ids) -> MediaSource`. The generic stage asks
only for the recordings the action grid contains and never branches on a name.

| Dataset | `recording_id` | Media | Offset |
| --- | --- | --- | --- |
| EgoCom | `video_info.video_name` | the POV video whose stem is the id (`240p/20min/<video_name>.MP4`); audio is embedded | 0 |
| Ego4D | AV `clip_uid` | the source video named by `clips_clean.video_uid` (`v2/video_540ss/<video_uid>.mp4`); audio is embedded | `(video_start_frame - clip_start_frame) / 30` — by frame index; the release's `video_start_sec` is on the full-scale video's timeline, not the 540ss file's |

Both mappings are the ones the native focal voice-state build already uses to
measure media coverage. A key matching more than one file of the same kind is
an error, never a silent choice.

## Validation

The build refuses to write a manifest that has:

- a null `dataset` or `recording_id`;
- a row with neither `video_path` nor `audio_path`;
- a duplicate `(dataset, recording_id)`;
- an absolute, home-relative, drive-letter, backslash or `..` path;
- a `recording_id` the action grid does not contain;
- media spread over more than one corpus directory.

## Coverage and strictness

The report counts, per dataset: recordings in the action grid and in the
manifest, video available, audio path available, video-only, audio-only,
both, video with embedded audio, and recordings without resolved media with
their reasons (`media_missing`, `clip_unknown`).

A canonical recording without media fails the build (`status: FAIL`, exit
code 2) when it still has labelled (`action_valid`) grid slots. That means the
grid holds labels the media cannot back. A recording the grid already masks
entirely (the native state marked its media missing) is listed but tolerated,
because it contributes no label.

## Physical check

When the raw corpus is mounted locally:

```bash
uv run conv-wm audit media-manifest --dataset all
```

resolves `raw root / corpus_directory / path` for every row and reports
missing files by relative path in
`${paths.reports}/media_manifest/<dataset>/file_check.json`. The resolved
location is never written back.

## Outputs and lineage

| Artifact | Path |
| --- | --- |
| manifest | `${paths.model_ready}/<dataset>/media_manifest.parquet` |
| report | `${paths.reports}/media_manifest/<dataset>/report.json` |

The report follows the other stages: git commit and dirty flag, command,
config checksum, `uv.lock` checksum, `media_schema_version`,
`lineage_chain`, and the checksums of the action grid, its report, the media
metadata table and the adapter tables. The model-ready build reads the
manifest when one exists. It refuses one whose checksum no longer matches its
report or that was built from a different action grid. It then adds a
path-free `media` section to `metadata.json` (file, sha256, schema, path
semantics, coverage, `raw_media_distributed: false`).

Raw media is never copied, uploaded or redistributed; the manifest only
references it.
