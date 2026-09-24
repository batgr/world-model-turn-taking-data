"""Ego4D AV camera-wearer ``voice_segments`` -> native focal voice recordings.

Semantics (v0): inside a camera-wearer voice segment the wearer is
``SPEAKING``; outside every valid wearer segment the wearer is ``SILENT``; a
clip flagged invalid by the release, an explicitly missing voice region, or
time the media does not cover is ``UNKNOWN``. Voice segments keep their native
AV convention: one annotated vocal episode may absorb short internal pauses
and is never split acoustically.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.datasets.ego4d_cleaning import RULE_VERSION
from conv_wm.data.datasets.ego4d_media import clip_media_offset_s
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
        media_unknown = _media_unknown(
            coverage.get(str(row["video_uid"])),
            clip_start_s=start_s,
            clip_end_s=end_s,
            media_offset_s=clip_media_offset_s(row),
        )
        if media_unknown and media_unknown[0].annotation_id != "media_uncovered":
            clips_without_media += 1
        unknown.extend(media_unknown)
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
        annotation_paths=(clips_path, persons_path, voice_path, missing_path),
        annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
        cleaning_rule_version=RULE_VERSION,
        statistics={
            "clips": len(clips),
            "clips_without_camera_wearer": clips_without_wearer,
            "invalid_clips": invalid_clips,
            "clips_with_missing_voice_regions": clips_with_missing_regions,
            "clips_without_usable_media": clips_without_media,
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


def _media_unknown(
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


__all__ = ["ANNOTATION_SCHEMA_VERSION", "load_ego4d_native_voice"]
