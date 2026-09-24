"""EgoCom speaker-attributed word timings -> native focal voice recordings.

Semantics (v0): for the wearer of one point-of-view video, every timed
transcript token attributed to that speaker is ``SPEAKING``; the rest of the
video is ``SILENT``; media the audit could not probe or that does not cover
the video is ``UNKNOWN``. Speech of other participants, including participants
without a point-of-view video, never changes the wearer's state.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.datasets.egocom_cleaning import RULE_VERSION
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

ANNOTATION_SCHEMA_VERSION = "egocom-ground_truth_transcriptions-v1"
"""EgoCom ``ground_truth_transcriptions.csv`` (word-level, speaker-attributed)."""


def media_unknown(
    coverage: MediaCoverage | None, end_s: float
) -> list[NativeAnnotation]:
    """Part time ``[0, end_s)`` the POV video's audio does not cover, as UNKNOWN."""
    if coverage is None:
        return [NativeAnnotation(0.0, end_s, "media_missing")]
    if not coverage.probe_ok:
        return [NativeAnnotation(0.0, end_s, "media_unprobed")]
    unknown: list[NativeAnnotation] = []
    if coverage.audio_start_s > 0.0:
        unknown.append(
            NativeAnnotation(0.0, min(coverage.audio_start_s, end_s), "media_uncovered")
        )
    if coverage.audio_end_s < end_s:
        unknown.append(
            NativeAnnotation(max(coverage.audio_end_s, 0.0), end_s, "media_uncovered")
        )
    return unknown


def load_egocom_native_voice(
    cfg: DictConfig, media: pd.DataFrame
) -> NativeFocalVoiceSource:
    """Map cleaned EgoCom video metadata and word timings to one recording per POV."""
    info_path = get_path(cfg, "egocom", "interim", "video_info_clean")
    transcript_path = get_path(cfg, "egocom", "interim", "ground_truth_clean")
    info = require_table(info_path)
    transcript = require_table(transcript_path)
    coverage = media_coverage_by_stem(media, "egocom")
    speaker = pd.to_numeric(transcript["speaker_id"], errors="coerce")
    transcript_by_part = {
        str(part): group
        for part, group in transcript.groupby("conversation_id", sort=False)
    }
    pov_speakers = {
        str(part): {int(value) for value in group["video_speaker_id"]}
        for part, group in info.groupby("conversation_id", sort=False)
    }
    recordings: list[NativeFocalRecording] = []
    videos_without_media = 0
    for row in info.sort_values("video_name").to_dict(orient="records"):
        view_id = str(row["video_name"])
        part = str(row["conversation_id"])
        wearer = int(row["video_speaker_id"])
        end_s = float(row["duration_seconds"])
        words = transcript_by_part.get(part, transcript.iloc[0:0])
        wearer_words = words.loc[speaker.reindex(words.index).eq(wearer)]
        speaking = timed_intervals(
            wearer_words, "ground_truth_clean", "startTime", "endTime"
        )
        media_row = coverage.get(view_id)
        if media_row is None or not media_row.probe_ok:
            videos_without_media += 1
        unknown = media_unknown(media_row, end_s)
        recordings.append(
            NativeFocalRecording(
                dataset="egocom",
                recording_id=view_id,
                sync_group_id=part,
                view_id=view_id,
                wearer_id=str(wearer),
                start_s=0.0,
                end_s=end_s,
                speaking=speaking,
                unknown=tuple(unknown),
                source_kind=SourceKind.EGOCOM_TRANSCRIPT,
                annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
            )
        )
    speakers_without_pov = sum(
        len(
            {int(v) for v in speaker.reindex(group.index).dropna()}
            - pov_speakers.get(part, set())
        )
        for part, group in transcript_by_part.items()
    )
    return NativeFocalVoiceSource(
        dataset="egocom",
        recordings=recordings,
        annotation_paths=(info_path, transcript_path),
        annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
        cleaning_rule_version=RULE_VERSION,
        statistics={
            "videos": len(info),
            "videos_without_usable_media": videos_without_media,
            "conversation_parts": len(pov_speakers),
            "transcript_speakers_without_pov": speakers_without_pov,
        },
        limitations=(
            (
                "SPEAKING is the union of the wearer's timed transcript tokens: a "
                "transcript-derived speaker interval, not a vocal-episode annotation; "
                "gaps between consecutive tokens are SILENT, however short."
            ),
            (
                "Absence of annotation is SILENT in v0 although the vocal annotation "
                "coverage audit measured a conditional focal-specific coverage of "
                "83.49 % on the EgoCom sub-population where it could be measured."
            ),
        ),
    )


__all__ = ["ANNOTATION_SCHEMA_VERSION", "load_egocom_native_voice", "media_unknown"]
