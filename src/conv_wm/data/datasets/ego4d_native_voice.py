"""Ego4D AV camera-wearer ``voice_segments`` -> native focal voice recordings.

Semantics (v0): inside a camera-wearer voice segment the wearer is
``SPEAKING``; outside every valid wearer segment the wearer is ``SILENT``; a
clip flagged invalid by the release, an explicitly missing voice region,
time the media does not cover, or an audio dropout of the source video is
``UNKNOWN``. Voice segments keep their native
AV convention: one annotated vocal episode may absorb short internal pauses
and is never split acoustically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path, pipeline_paths
from conv_wm.data.datasets.ego4d_cleaning import RULE_VERSION
from conv_wm.data.datasets.ego4d_media import clip_media_offset_s
from conv_wm.data.pipeline_inputs import require_file
from conv_wm.data.vocal.native_source import (
    MediaCoverage,
    NativeFocalVoiceSource,
    media_coverage_by_stem,
    require_table,
    timed_intervals,
)
from conv_wm.data.vocal.native_state import (
    NativeAnnotation,
    NativeFocalRecording,
    SourceKind,
)

ANNOTATION_SCHEMA_VERSION = "ego4d-av-v2-voice_segments"
"""Ego4D v2 AV benchmark release (``av_train.json`` / ``av_val.json``)."""

AUDIO_TIMELINE_DIR = Path("temporal") / "audio_timeline"
"""Where ``conv-wm audit audio`` writes its packet-timeline tables."""
AUDIO_DROPOUT_MIN_S = 0.100
"""Shortest dropout declared UNKNOWN: the model's media reader fills every
decoded PTS gap at least this long with silence, so the audio there is not
the recording's."""


def load_ego4d_native_voice(
    cfg: DictConfig, media: pd.DataFrame
) -> NativeFocalVoiceSource:
    """Map cleaned Ego4D clips, persons and voice segments to one recording per clip."""
    clips_path = get_path(cfg, "ego4d", "interim", "clips_clean")
    persons_path = get_path(cfg, "ego4d", "interim", "persons_clean")
    voice_path = get_path(cfg, "ego4d", "interim", "voice_segments_clean")
    missing_path = get_path(cfg, "ego4d", "interim", "missing_voice_segments_clean")
    clips = require_table(clips_path)
    persons = require_table(persons_path)
    voice = require_table(voice_path)
    missing = require_table(missing_path)
    coverage = media_coverage_by_stem(media, "ego4d")
    dropouts = load_audio_dropouts(cfg, coverage)

    wearer_by_clip = {
        str(row["clip_uid"]): str(row["person_id"])
        for row in persons.loc[persons["is_camera_wearer"].astype(bool)].to_dict(
            orient="records"
        )
    }
    voice_by_clip = {
        str(uid): group for uid, group in voice.groupby("clip_uid", sort=False)
    }
    missing_by_clip = {
        str(uid): group for uid, group in missing.groupby("clip_uid", sort=False)
    }
    recordings: list[NativeFocalRecording] = []
    clips_without_wearer = 0
    invalid_clips = 0
    clips_with_missing_regions = 0
    clips_without_media = 0
    clips_with_audio_dropouts = 0
    for row in clips.sort_values("clip_uid").to_dict(orient="records"):
        clip_uid = str(row["clip_uid"])
        wearer_id = wearer_by_clip.get(clip_uid)
        if wearer_id is None:
            clips_without_wearer += 1
            continue
        start_s = float(row["clip_start_sec"])
        end_s = float(row["clip_end_sec"])
        clip_voice = voice_by_clip.get(clip_uid, voice.iloc[0:0])
        speaking = timed_intervals(
            clip_voice.loc[clip_voice["person_id"].astype(str).eq(wearer_id)],
            "voice_segments_clean",
            "start_time",
            "end_time",
        )
        unknown: list[NativeAnnotation] = []
        if not bool(row["valid"]):
            invalid_clips += 1
            unknown.append(NativeAnnotation(start_s, end_s, "clip_invalid"))
        clip_missing = missing_by_clip.get(clip_uid, missing.iloc[0:0])
        wearer_missing = clip_missing.loc[
            clip_missing["person_id"].isna()
            | clip_missing["person_id"].astype(str).eq(wearer_id)
        ]
        missing_regions = timed_intervals(
            wearer_missing, "missing_voice_segments_clean", "start_time", "end_time"
        )
        if missing_regions:
            clips_with_missing_regions += 1
            unknown.extend(missing_regions)
        media_unknown_regions = media_unknown(
            coverage.get(str(row["video_uid"])),
            clip_start_s=start_s,
            clip_end_s=end_s,
            media_offset_s=clip_media_offset_s(row),
        )
        if (
            media_unknown_regions
            and media_unknown_regions[0].annotation_id != "media_uncovered"
        ):
            clips_without_media += 1
        unknown.extend(media_unknown_regions)
        dropout_regions = dropout_unknown(
            dropouts.get(str(row["video_uid"]), ()),
            clip_start_s=start_s,
            clip_end_s=end_s,
            media_offset_s=clip_media_offset_s(row),
        )
        if dropout_regions:
            clips_with_audio_dropouts += 1
        unknown.extend(dropout_regions)
        recordings.append(
            NativeFocalRecording(
                dataset="ego4d",
                recording_id=clip_uid,
                sync_group_id=None,
                view_id=str(row["video_uid"]),
                wearer_id=wearer_id,
                start_s=start_s,
                end_s=end_s,
                speaking=speaking,
                unknown=tuple(unknown),
                source_kind=SourceKind.EGO4D_VOICE_SEGMENTS,
                annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
            )
        )
    return NativeFocalVoiceSource(
        dataset="ego4d",
        recordings=recordings,
        annotation_paths=(
            clips_path,
            persons_path,
            voice_path,
            missing_path,
            *audio_timeline_tables(cfg),
        ),
        annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
        cleaning_rule_version=RULE_VERSION,
        statistics={
            "clips": len(clips),
            "clips_without_camera_wearer": clips_without_wearer,
            "invalid_clips": invalid_clips,
            "clips_with_missing_voice_regions": clips_with_missing_regions,
            "clips_without_usable_media": clips_without_media,
            "clips_with_audio_dropouts": clips_with_audio_dropouts,
        },
        limitations=(
            (
                "SPEAKING is the native AV3 vocal-episode state: one camera-wearer "
                "voice segment may absorb short internal pauses; boundaries are the "
                "official ones and are never refined acoustically."
            ),
            (
                "Wearer speech the annotators did not mark is SILENT in v0; no "
                "independent focal-specific completeness measurement exists for Ego4D "
                "(vocal annotation coverage audit)."
            ),
        ),
    )


def media_unknown(
    coverage: MediaCoverage | None,
    *,
    clip_start_s: float,
    clip_end_s: float,
    media_offset_s: float,
) -> list[NativeAnnotation]:
    """Clip time the source video's audio does not cover, on the clip timeline."""
    if coverage is None:
        return [NativeAnnotation(clip_start_s, clip_end_s, "media_missing")]
    if not coverage.probe_ok:
        return [NativeAnnotation(clip_start_s, clip_end_s, "media_unprobed")]
    covered_start = coverage.audio_start_s - media_offset_s
    covered_end = coverage.audio_end_s - media_offset_s
    output: list[NativeAnnotation] = []
    if covered_start > clip_start_s:
        output.append(
            NativeAnnotation(
                clip_start_s, min(covered_start, clip_end_s), "media_uncovered"
            )
        )
    if covered_end < clip_end_s:
        output.append(
            NativeAnnotation(
                max(covered_end, clip_start_s), clip_end_s, "media_uncovered"
            )
        )
    return output


def audio_timeline_tables(cfg: DictConfig) -> tuple[Path, Path]:
    """The audio audit's per-file and per-event tables the dropouts come from."""
    root = pipeline_paths(cfg).reports / AUDIO_TIMELINE_DIR
    return (
        root / "audio_packet_timeline_files.parquet",
        root / "audio_packet_timeline_events.parquet",
    )


def load_audio_dropouts(
    cfg: DictConfig, coverage: dict[str, MediaCoverage]
) -> dict[str, list[tuple[float, float]]]:
    """Audio dropouts of at least ``AUDIO_DROPOUT_MIN_S`` per source video.

    Intervals are on the media timeline of ``media_unknown``. A dropout event
    is stamped at the PTS of the packet after the jump; the decoded audio
    stops ``dropout_duration_samples`` earlier. Every Ego4D video whose audio
    was probed must have a measured packet timeline, so no dropout goes
    undeclared because the audit is stale or failed on a file.
    """
    command = "conv-wm audit audio"
    files_path, events_path = audio_timeline_tables(cfg)
    files = pd.read_parquet(require_file(files_path, command))
    events = pd.read_parquet(require_file(events_path, command))
    files = files.loc[files["dataset"].eq("ego4d")]
    measured = {
        Path(str(path)).stem
        for path in files.loc[files["timeline_status"].eq("measured"), "relative_path"]
    }
    unmeasured = sorted(
        stem
        for stem, item in coverage.items()
        if item.probe_ok and stem not in measured
    )
    if unmeasured:
        raise ValueError(
            f"{len(unmeasured)} Ego4D videos have no measured audio timeline "
            f"(first: {unmeasured[:3]}); rerun `{command}`"
        )
    rows = events.loc[
        events["dataset"].eq("ego4d")
        & events["event_category"].eq("audio_dropout")
        & (events["dropout_duration_samples"] / events["sample_rate_hz"]).ge(
            AUDIO_DROPOUT_MIN_S
        )
    ]
    output: dict[str, list[tuple[float, float]]] = {}
    for row in rows.to_dict(orient="records"):
        end = float(row["event_time_sec"])
        duration = float(row["dropout_duration_samples"]) / int(row["sample_rate_hz"])
        if not math.isfinite(end) or not math.isfinite(duration):
            raise ValueError(f"Non-finite audio dropout in {row['relative_path']}")
        output.setdefault(Path(str(row["relative_path"])).stem, []).append(
            (end - duration, end)
        )
    return output


def dropout_unknown(
    dropouts: Sequence[tuple[float, float]],
    *,
    clip_start_s: float,
    clip_end_s: float,
    media_offset_s: float,
) -> list[NativeAnnotation]:
    """The part of each media-time dropout inside the clip, on the clip timeline."""
    output = []
    for media_start, media_end in dropouts:
        start = max(media_start - media_offset_s, clip_start_s)
        end = min(media_end - media_offset_s, clip_end_s)
        if end > start:
            output.append(NativeAnnotation(start, end, "media_audio_dropout"))
    return output


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "AUDIO_DROPOUT_MIN_S",
    "audio_timeline_tables",
    "dropout_unknown",
    "load_audio_dropouts",
    "load_ego4d_native_voice",
    "media_unknown",
]
