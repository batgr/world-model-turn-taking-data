"""Ego4D AV benchmark: dataset-specific knowledge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera import Check

from conv_wm.data.datasets.spec import (
    AudioInterpretation,
    DatasetSpec,
    RelationSpec,
    StructuralSpec,
    TableSpec,
    parquet_table,
)
from conv_wm.data.media.audio.decode import DecodeValidationCase
from conv_wm.data.media.audio.interpretation import KnownBoundaryGrid
from conv_wm.data.media.audio.summary import quantiles

EGO4D_CLIPS_SCHEMA = pa.DataFrameSchema(
    {
        "split": pa.Column(str, nullable=False, checks=Check.isin(["train", "val"])),
        "clip_uid": pa.Column(str, nullable=False, unique=True),
        "source_clip_uid": pa.Column(str, nullable=False),
        "video_uid": pa.Column(str, nullable=False),
        "video_start_sec": pa.Column(float, nullable=False),
        "video_end_sec": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(int, nullable=False),
        "video_end_frame": pa.Column(int, nullable=False),
        "clip_start_sec": pa.Column(int, nullable=False),
        "clip_end_sec": pa.Column(float, nullable=False),
        "clip_start_frame": pa.Column(int, nullable=False),
        "clip_end_frame": pa.Column(int, nullable=False),
    },
    strict=True,
    name="ego4d_clips_clean",
)


EGO4D_PERSONS_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "person_id": pa.Column(str, nullable=False),
        "is_camera_wearer": pa.Column(bool, nullable=False),
    },
    unique=["clip_uid", "person_id"],
    strict=True,
    name="ego4d_persons_clean",
)


EGO4D_TRACKING_PATHS_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "person_id": pa.Column(str, nullable=False),
        "track_id": pa.Column(str, nullable=False),
        "unmapped_frames_count": pa.Column(
            int,
            nullable=False,
            checks=Check.ge(0),
        ),
        "unmapped_frames": pa.Column(object, nullable=False),
    },
    unique=["clip_uid", "person_id", "track_id"],
    strict=True,
    name="ego4d_tracking_paths_clean",
)


EGO4D_TRACKS_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "person_id": pa.Column(str, nullable=False),
        "track_id": pa.Column(str, nullable=False),
        "x": pa.Column(float, nullable=False),
        "y": pa.Column(float, nullable=False),
        "width": pa.Column(float, nullable=False, checks=Check.gt(0)),
        "height": pa.Column(float, nullable=False, checks=Check.gt(0)),
        "clip_frame": pa.Column(int, nullable=False),
        "video_frame": pa.Column(int, nullable=False),
    },
    unique=["clip_uid", "person_id", "track_id", "clip_frame"],
    strict=True,
    name="ego4d_tracks_clean",
)


EGO4D_VOICE_SEGMENTS_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "start_time": pa.Column(float, nullable=False),
        "end_time": pa.Column(float, nullable=False),
        "start_frame": pa.Column(float, nullable=False),
        "end_frame": pa.Column(float, nullable=False),
        "video_start_time": pa.Column(float, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(float, nullable=False),
        "video_end_frame": pa.Column(float, nullable=False),
        "person_id": pa.Column(str, nullable=False),
    },
    strict=True,
    name="ego4d_voice_segments_clean",
)


EGO4D_TRANSCRIPTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "transcription": pa.Column(str, nullable=False),
        "start_time_sec": pa.Column(float, nullable=False),
        "end_time_sec": pa.Column(float, nullable=False),
        "person_id": pa.Column(str, nullable=False),
        "video_start_time": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(float, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_end_frame": pa.Column(float, nullable=False),
    },
    strict=True,
    name="ego4d_transcriptions_clean",
)


EGO4D_SOCIAL_SEGMENTS_TALKING_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "start_time": pa.Column(float, nullable=False),
        "end_time": pa.Column(float, nullable=False),
        "start_frame": pa.Column(int, nullable=False),
        "end_frame": pa.Column(int, nullable=False),
        "video_start_time": pa.Column(float, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(int, nullable=False),
        "video_end_frame": pa.Column(int, nullable=False),
        "person": pa.Column(str, nullable=False),
        "target": pa.Column(str, nullable=False),
        "is_at_me": pa.Column(bool, nullable=False),
    },
    strict=True,
    name="ego4d_social_segments_talking_clean",
)


EGO4D_SOCIAL_SEGMENTS_LOOKING_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "start_time": pa.Column(float, nullable=False),
        "end_time": pa.Column(float, nullable=False),
        "start_frame": pa.Column(float, nullable=False),
        "end_frame": pa.Column(float, nullable=False),
        "video_start_time": pa.Column(float, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(float, nullable=False),
        "video_end_frame": pa.Column(float, nullable=False),
        "person": pa.Column(str, nullable=False),
    },
    strict=True,
    name="ego4d_social_segments_looking_clean",
)


def _clip_relation(table: str, short: str) -> RelationSpec:
    return RelationSpec(
        name=f"{short}_clip_uid_in_clips",
        child_table=table,
        child_columns=("clip_uid",),
        parent_table="clips_clean",
        parent_columns=("clip_uid",),
    )


STRUCTURE = StructuralSpec(
    tables=tuple(
        TableSpec(
            name=name, schema=schema, load=parquet_table("ego4d", "interim", name)
        )
        for name, schema in (
            ("clips_clean", EGO4D_CLIPS_SCHEMA),
            ("persons_clean", EGO4D_PERSONS_SCHEMA),
            ("tracking_paths_clean", EGO4D_TRACKING_PATHS_SCHEMA),
            ("tracks_clean", EGO4D_TRACKS_SCHEMA),
            ("voice_segments_clean", EGO4D_VOICE_SEGMENTS_SCHEMA),
            ("transcriptions_clean", EGO4D_TRANSCRIPTIONS_SCHEMA),
            ("social_segments_talking_clean", EGO4D_SOCIAL_SEGMENTS_TALKING_SCHEMA),
            ("social_segments_looking_clean", EGO4D_SOCIAL_SEGMENTS_LOOKING_SCHEMA),
        )
    ),
    relations=(
        _clip_relation("persons_clean", "persons"),
        _clip_relation("tracking_paths_clean", "tracking_paths"),
        _clip_relation("tracks_clean", "tracks"),
        _clip_relation("voice_segments_clean", "voice_segments"),
        _clip_relation("transcriptions_clean", "transcriptions"),
        _clip_relation("social_segments_talking_clean", "social_talking"),
        _clip_relation("social_segments_looking_clean", "social_looking"),
        RelationSpec(
            name="tracking_paths_person_in_persons",
            child_table="tracking_paths_clean",
            child_columns=("clip_uid", "person_id"),
            parent_table="persons_clean",
            parent_columns=("clip_uid", "person_id"),
        ),
        RelationSpec(
            name="tracks_path_in_tracking_paths",
            child_table="tracks_clean",
            child_columns=("clip_uid", "person_id", "track_id"),
            parent_table="tracking_paths_clean",
            parent_columns=("clip_uid", "person_id", "track_id"),
        ),
        RelationSpec(
            name="voice_segments_person_in_persons",
            child_table="voice_segments_clean",
            child_columns=("clip_uid", "person_id"),
            parent_table="persons_clean",
            parent_columns=("clip_uid", "person_id"),
        ),
    ),
)

STITCH_PERIOD_SEC = 300.0
"""Ego4D recordings are joined every 300 s; audio events cluster on that grid."""

SYSTEMATIC_PROFILE_SAMPLE_RATE = 44100
"""Regime whose dense short/long PTS cadence deserves a full decoded profile."""


def systematic_profile_cases(files: pd.DataFrame) -> list[DecodeValidationCase]:
    """Four decoded windows per 44.1 kHz file: start, middle, drift peak, end."""
    cases: list[DecodeValidationCase] = []
    subset = files.loc[files["sample_rate_hz"].eq(SYSTEMATIC_PROFILE_SAMPLE_RATE)]
    for _, row in subset.sort_values("relative_path").iterrows():
        file_id = Path(str(row["relative_path"])).stem[:8]
        duration = float(row["audio_duration_sec"])
        for window, event_time in (
            ("start", 0.0),
            ("middle", duration / 2.0),
            ("max_offset", float(row["max_abs_cumulative_pcm_drift_time_sec"])),
            ("end", duration),
        ):
            cases.append(
                DecodeValidationCase(
                    case=f"ego4d_44100_{file_id}_{window}",
                    selection_reason="ego4d_44100_systematic_profile",
                    dataset="ego4d",
                    relative_path=str(row["relative_path"]),
                    event_time_sec=event_time,
                )
            )
    return cases


def characterize_44100_regime(
    files: pd.DataFrame, decode_validation: pd.DataFrame
) -> dict[str, object]:
    """Describe the two observed 44.1 kHz timing regimes without correcting them."""
    subset = files.loc[files["sample_rate_hz"].eq(SYSTEMATIC_PROFILE_SAMPLE_RATE)]
    dense = subset.loc[
        subset["robust_nominal_packet_duration_samples"]
        < 0.95 * subset["reference_decoded_samples"]
    ]
    sparse = subset.drop(index=dense.index)
    decoded = (
        decode_validation.loc[
            decode_validation["selection_reason"].eq("ego4d_44100_systematic_profile")
        ]
        if not decode_validation.empty
        else pd.DataFrame()
    )
    return {
        "n_files": len(subset),
        "n_dense_compensating_cadence_files": len(dense),
        "n_sparse_near_reference_files": len(sparse),
        "dense_robust_nominal_duration_samples": quantiles(
            dense["robust_nominal_packet_duration_samples"]
        ),
        "sparse_robust_nominal_duration_samples": quantiles(
            sparse["robust_nominal_packet_duration_samples"]
        ),
        "final_pcm_drift_ms_quantiles": quantiles(
            subset["final_cumulative_pcm_drift_ms"]
        ),
        "max_abs_cumulative_pcm_drift_ms_quantiles": quantiles(
            subset["max_abs_cumulative_pcm_drift_ms"]
        ),
        "targeted_decode": {
            "n_files": int(decoded["relative_path"].nunique())
            if not decoded.empty
            else 0,
            "n_windows": len(decoded),
            "n_frames": (
                int(decoded["n_valid_decoded_sample_counts"].fillna(0).sum())
                if not decoded.empty
                else 0
            ),
            "n_non_reference_frames": (
                int(decoded["n_non_reference_decoded_frames"].fillna(0).sum())
                if not decoded.empty
                else 0
            ),
        },
        "interpretation": (
            "Dense files use compensating short/long PTS increments; sparse files "
            "are predominantly reference-sized steps. Decoded frames stay at the "
            "reference size, so no correction is inferred. Native PTS remains the clock."
        ),
    }


EGO4D = DatasetSpec(
    name="ego4d",
    description="Ego4D v2 audio-visual diarization benchmark clips (video_540ss).",
    structure=STRUCTURE,
    audio=AudioInterpretation(
        known_boundary_grid=KnownBoundaryGrid(period_sec=STITCH_PERIOD_SEC),
        extra_decode_cases=systematic_profile_cases,
        summary_section=characterize_44100_regime,
    ),
)
