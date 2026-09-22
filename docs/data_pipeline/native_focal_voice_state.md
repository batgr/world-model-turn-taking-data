# Native focal voice state (v0)

## What it is

`conv-wm build native-focal-voice-state` turns the native annotations of each
dataset into one continuous, exhaustive timeline of the wearer's vocal state
per point-of-view recording:

```text
native annotations  ->  canonical focal vocal-state intervals
                        SPEAKING | SILENT | UNKNOWN
```

The prototype is ego-only: for every view, `actor_id = wearer_id`. Nothing
acoustic is involved — no voice activity detector, no speaker model, no
completion of the annotations. Timestamps are continuous seconds on the
dataset's canonical timeline; the 100 ms decision grid and the
`NO_EVENT` / `ONSET` / `OFFSET` actions are a later step that consumes this
layer.

This is a **native-annotation-derived vocal state**, not an exhaustive
physical vocal-activity ground truth. The two corpora do not share the same
temporal semantics, and the layer does not hide it:

| Dataset | `SPEAKING` means | `SILENT` means | `UNKNOWN` means |
| --- | --- | --- | --- |
| Ego4D | inside an official AV `voice_segments` interval of the camera wearer — a native vocal *episode*, which may absorb short internal pauses | outside every wearer voice segment | clip flagged `valid = false`, an entry of the release's `missing_voice_segments` (for the wearer or unattributed), or clip time the source video's audio does not cover |
| EgoCom | inside a timed transcript token attributed to the wearer — a transcript-derived speaker interval; gaps between consecutive tokens are `SILENT`, however short | outside every timed wearer token | media the audit could not probe, or video time its audio does not cover |

Boundaries are the official ones. Ego4D segments are never split or refined
acoustically; EgoCom tokens are never bridged. Speech of other participants —
including EgoCom participants without a point-of-view video — never changes
the wearer's state.

## Accepted limitations (documented, not corrected in v0)

- **EgoCom**: absence of annotation is `SILENT`, although the vocal
  annotation coverage audit measured a conditional focal-specific annotation
  coverage of **83.49 %** on the 150/175 recordings where it could be measured
  ([`vocal_annotation_coverage.md`](vocal_annotation_coverage.md)). The
  report references that audit's artifact, version and checksum; it does not
  recompute it.
- **Ego4D**: `voice_segments` follow the native AV3 convention and can
  include short internal pauses in one `SPEAKING` episode. No independent
  focal-specific completeness measurement exists for Ego4D.
- **EgoCom token gaps**: consecutive wearer tokens usually abut, but the
  union of word intervals produces many `SILENT` gaps shorter than the future
  100 ms step (`short_silent_interval_count` in the report). Handling them is
  the control layer's decision, not this layer's: they are bridged there
  ([`control_focal_voice_state.md`](control_focal_voice_state.md)) and this
  timeline keeps them.

Both limitations belong in the Dataset Card.

## Inputs

- cleaned interim tables written by `conv-wm clean annotations` — Ego4D:
  `clips_clean` (with the release `valid` flag), `persons_clean`,
  `voice_segments_clean`, `missing_voice_segments_clean`; EgoCom:
  `video_info_clean`, `ground_truth_clean`;
- the media metadata audit (`conv-wm audit media`), which decides whether a
  recording's audio exists and which part of the annotated window it covers;
- the raw manifest (`conv-wm audit manifest`), checksummed into the report.

A missing prerequisite stops the command and names the command that produces
it.

## Timeline construction

For one recording the adapter provides the nominal annotated window, the
wearer's native `SPEAKING` intervals and the `UNKNOWN` regions it declares.
`build_native_timeline` then cuts the window at every interval boundary and
labels each piece with a fixed precedence — `UNKNOWN` over `SPEAKING` over
`SILENT` — so an untrustworthy region is never `SILENT`, and merges adjacent
pieces only when state, source kind and source annotation ids are identical.
A region covered by two overlapping native segments keeps both ids (joined by
`|`). The result covers the window exactly, is sorted, has no gap and no
overlap; tests enforce all of it.

## Output schema

`${paths.processed}/native_focal_voice_state/<dataset>/focal_voice_intervals.parquet`

| column | meaning |
| --- | --- |
| `dataset`, `recording_id`, `sync_group_id`, `view_id`, `wearer_id` | Ego4D: clip, no sync group, source video, camera wearer; EgoCom: POV video, conversation part, POV video, speaker of the POV |
| `canonical_start_s`, `canonical_end_s` | continuous seconds on the canonical timeline (Ego4D clip-relative, EgoCom conversation-part-relative) |
| `voice_state` | `SPEAKING`, `SILENT`, `UNKNOWN` |
| `source_kind` | `ego4d_voice_segments`, `egocom_transcript`, `invalid_or_missing` |
| `source_annotation_id` | `<clean table>#<row index>` of the native rows (joined by `|`), a declared reason for `UNKNOWN` (`clip_invalid`, `missing_voice_segments_clean#<row>`, `media_missing`, `media_unprobed`, `media_uncovered`), null for `SILENT` |
| `annotation_schema_version` | `ego4d-av-v2-voice_segments` / `egocom-ground_truth_transcriptions-v1` |
| `native_state_schema_version` | this layer's schema version |

No confidence, score or method column exists: there is nothing inferred.

## Reports

`${paths.reports}/native_focal_voice_state/<dataset>/`:

- `summary.parquet` — one row per recording: window, valid, speaking, silent
  and unknown durations, ratios, interval counts (including silent intervals
  shorter than 100 ms);
- `report.json` — per-dataset statistics (recording count, valid duration,
  state durations and ratios, native `SPEAKING` interval count, Ego4D clips
  with invalid or missing regions, EgoCom speakers without a POV), the state
  semantics and limitations, and the lineage: schema versions, `git_commit`,
  `git_dirty`, command, `created_at`, SHA-256 of the manifest, the media
  metadata table, every native annotation table, `uv.lock` and the output
  Parquet files, the source release and cleaning-rule versions, and the
  reference to the coverage audit artifact.

The report answers: which native annotations produced this timeline, with
which code revision, which schema, and from which files.

## Run

```bash
uv run conv-wm clean annotations
uv run conv-wm audit media                      # once, after the manifest
uv run conv-wm build native-focal-voice-state --dataset all
uv run conv-wm build native-focal-voice-state --dataset egocom
```

The build is deterministic and takes seconds; it needs no optional extra.
