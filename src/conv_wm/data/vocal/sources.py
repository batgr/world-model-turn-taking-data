"""Dataset-specific construction of focal-recording audit inputs.

This is the only module that knows how EgoCom and Ego4D tables map to a
``FocalRecording``. Generic VAD, identity and coverage code remains independent
of dataset schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path, pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.vocal.intervals import clip
from conv_wm.data.vocal.records import AudioSource, FocalRecording


@dataclass(frozen=True)
class DatasetRecordingSet:
    """Recordings plus the input provenance and identity policy of one corpus."""

    dataset: str
    recordings: list[FocalRecording]
    annotation_paths: tuple[Path, ...]
    identity_methods: tuple[str, ...]
    annotation_provenance: str
    limitations: tuple[str, ...]


def _required_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with="conv-wm clean annotations")
    return pd.read_parquet(path)


def _media_index(media: pd.DataFrame, dataset: str) -> dict[str, dict[str, Any]]:
    rows = media.loc[media["dataset"].eq(dataset) & media["probe_ok"]]
    return {
        Path(str(row["relative_path"])).stem: {
            str(key): value for key, value in row.items()
        }
        for row in rows.to_dict(orient="records")
    }


def _intervals(
    table: pd.DataFrame,
    start: str,
    end: str,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Unique finite table intervals clipped to the audited canonical window."""
    if table.empty:
        return np.empty((0, 2), dtype=float)
    values = (
        table[[start, end]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    return clip(values, lower, upper)


def load_egocom_recordings(cfg: DictConfig, media: pd.DataFrame) -> DatasetRecordingSet:
    """Map EgoCom POV metadata and word timings to one row per wearer/view."""
    info_path = get_path(cfg, "egocom", "interim", "video_info_clean")
    transcript_path = get_path(cfg, "egocom", "interim", "ground_truth_clean")
    info = _required_table(info_path)
    transcript = _required_table(transcript_path)
    media_by_stem = _media_index(media, "egocom")
    raw_root = pipeline_paths(cfg).raw
    info_by_group = {
        str(group_id): group.sort_values("video_name")
        for group_id, group in info.groupby("conversation_id", sort=True)
    }
    transcript_by_group = {
        str(group_id): group
        for group_id, group in transcript.groupby("conversation_id", sort=False)
    }
    recordings: list[FocalRecording] = []
    for row in info.sort_values("video_name").to_dict(orient="records"):
        view_id = str(row["video_name"])
        media_row = media_by_stem.get(view_id)
        if media_row is None:
            continue
        audio_start = float(media_row["audio_start_time_sec"])
        audio_end = audio_start + float(media_row["audio_duration_sec"])
        canonical_start = max(0.0, audio_start)
        canonical_end = min(float(row["duration_seconds"]), audio_end)
        if canonical_end <= canonical_start:
            continue
        group_id = str(row["conversation_id"])
        wearer_id = str(int(row["video_speaker_id"]))
        timed = transcript_by_group.get(group_id, pd.DataFrame())
        if timed.empty:
            focal_rows = other_rows = timed
        else:
            speaker = pd.to_numeric(timed["speaker_id"], errors="coerce")
            focal_rows = timed.loc[speaker.eq(int(wearer_id))]
            other_rows = timed.loc[speaker.ne(int(wearer_id))]
        focal = _intervals(
            focal_rows,
            "startTime",
            "endTime",
            lower=canonical_start,
            upper=canonical_end,
        )
        others = _intervals(
            other_rows,
            "startTime",
            "endTime",
            lower=canonical_start,
            upper=canonical_end,
        )
        other_views: list[AudioSource] = []
        for other in info_by_group[group_id].to_dict(orient="records"):
            other_view = str(other["video_name"])
            if other_view == view_id:
                continue
            other_media = media_by_stem.get(other_view)
            if other_media is None:
                continue
            other_start = float(other_media["audio_start_time_sec"])
            other_end = other_start + float(other_media["audio_duration_sec"])
            shared_start = max(canonical_start, other_start)
            shared_end = min(canonical_end, other_end)
            if shared_end <= shared_start:
                continue
            other_views.append(
                AudioSource(
                    path=str(raw_root / str(other_media["relative_path"])),
                    start_s=shared_start,
                    duration_s=shared_end - shared_start,
                    canonical_offset_s=shared_start,
                    view_id=other_view,
                )
            )
        recordings.append(
            FocalRecording(
                dataset="egocom",
                recording_id=view_id,
                view_id=view_id,
                wearer_id=wearer_id,
                sync_group_id=group_id,
                audio=AudioSource(
                    path=str(raw_root / str(media_row["relative_path"])),
                    start_s=canonical_start,
                    duration_s=canonical_end - canonical_start,
                    canonical_offset_s=canonical_start,
                    view_id=view_id,
                ),
                canonical_start_s=canonical_start,
                canonical_end_s=canonical_end,
                focal_annotation=focal,
                other_annotation=others,
                annotation_available=True,
                annotation_source=(
                    "ground_truth_clean.parquet; human-observed word timings on "
                    "the conversation-part timeline"
                ),
                other_views=tuple(
                    sorted(other_views, key=lambda source: source.view_id)
                ),
            )
        )
    return DatasetRecordingSet(
        dataset="egocom",
        recordings=recordings,
        annotation_paths=(info_path, transcript_path),
        identity_methods=(
            "focal_annotation",
            "other_annotation",
            "multi_device_energy",
        ),
        annotation_provenance="human_observed",
        limitations=(
            "Relative microphone energy is an attribution candidate, not ground truth.",
            (
                "Its fixed rule is used only when apparent per-recording precision on "
                "available focal and other-speaker annotations reaches the configured gate."
            ),
            "One global cross-correlation lag does not model residual device-clock drift.",
        ),
    )


def load_ego4d_recordings(cfg: DictConfig, media: pd.DataFrame) -> DatasetRecordingSet:
    """Map Ego4D clips and per-person voice segments to clip-relative windows."""
    clips_path = get_path(cfg, "ego4d", "interim", "clips_clean")
    persons_path = get_path(cfg, "ego4d", "interim", "persons_clean")
    voice_path = get_path(cfg, "ego4d", "interim", "voice_segments_clean")
    clips = _required_table(clips_path)
    persons = _required_table(persons_path)
    voice = _required_table(voice_path)
    media_by_stem = _media_index(media, "ego4d")
    raw_root = pipeline_paths(cfg).raw
    wearer_by_clip = {
        str(row["clip_uid"]): str(row["person_id"])
        for row in persons.loc[persons["is_camera_wearer"]].to_dict(orient="records")
    }
    voice_by_clip = {
        str(clip_uid): group
        for clip_uid, group in voice.groupby("clip_uid", sort=False)
    }
    recordings: list[FocalRecording] = []
    for row in clips.sort_values("clip_uid").to_dict(orient="records"):
        clip_uid = str(row["clip_uid"])
        video_uid = str(row["video_uid"])
        media_row = media_by_stem.get(video_uid)
        wearer_id = wearer_by_clip.get(clip_uid)
        if media_row is None or wearer_id is None:
            continue
        media_start = float(media_row["audio_start_time_sec"])
        media_end = media_start + float(media_row["audio_duration_sec"])
        requested_start = float(row["video_start_sec"])
        requested_end = float(row["video_end_sec"])
        source_start = max(requested_start, media_start)
        source_end = min(requested_end, media_end)
        if source_end <= source_start:
            continue
        canonical_start = source_start - requested_start
        canonical_end = source_end - requested_start
        clip_voice = voice_by_clip.get(clip_uid, pd.DataFrame())
        if clip_voice.empty:
            focal_rows = other_rows = clip_voice
        else:
            focal_rows = clip_voice.loc[
                clip_voice["person_id"].astype(str).eq(wearer_id)
            ]
            other_rows = clip_voice.loc[
                clip_voice["person_id"].astype(str).ne(wearer_id)
            ]
        focal = _intervals(
            focal_rows,
            "start_time",
            "end_time",
            lower=canonical_start,
            upper=canonical_end,
        )
        others = _intervals(
            other_rows,
            "start_time",
            "end_time",
            lower=canonical_start,
            upper=canonical_end,
        )
        recordings.append(
            FocalRecording(
                dataset="ego4d",
                recording_id=clip_uid,
                view_id=video_uid,
                wearer_id=wearer_id,
                sync_group_id=None,
                audio=AudioSource(
                    path=str(raw_root / str(media_row["relative_path"])),
                    start_s=source_start,
                    duration_s=source_end - source_start,
                    canonical_offset_s=canonical_start,
                    view_id=clip_uid,
                ),
                canonical_start_s=canonical_start,
                canonical_end_s=canonical_end,
                focal_annotation=focal,
                other_annotation=others,
                annotation_available=True,
                annotation_source=(
                    "voice_segments_clean.parquet joined to persons_clean.parquet; "
                    "human-observed per-person intervals on the clip timeline"
                ),
            )
        )
    return DatasetRecordingSet(
        dataset="ego4d",
        recordings=recordings,
        annotation_paths=(clips_path, persons_path, voice_path),
        identity_methods=("focal_annotation", "other_annotation"),
        annotation_provenance="human_observed",
        limitations=(
            "The mixed audio has no independent wearer-identity signal in the available data.",
            (
                "Annotation overlap can identify already annotated speakers but cannot prove "
                "the identity of unannotated acoustic activity."
            ),
            (
                "Focal-specific completeness is therefore unresolved; acoustic-only "
                "quantities remain valid."
            ),
        ),
    )


def load_recording_sets(
    cfg: DictConfig, media: pd.DataFrame, datasets: tuple[str, ...]
) -> list[DatasetRecordingSet]:
    """Load selected corpus adapters in deterministic dataset order."""
    loaders = {"ego4d": load_ego4d_recordings, "egocom": load_egocom_recordings}
    unknown = set(datasets) - set(loaders)
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    return [loaders[name](cfg, media) for name in sorted(datasets)]


__all__ = [
    "DatasetRecordingSet",
    "load_ego4d_recordings",
    "load_egocom_recordings",
    "load_recording_sets",
]
