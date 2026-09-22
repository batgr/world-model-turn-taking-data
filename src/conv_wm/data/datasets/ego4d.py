"""Ego4D AV benchmark: dataset-specific knowledge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera import Check

from conv_wm.data.annotations import (
    AnnotationProvenance,
    AnnotationScope,
    AnnotationSourceSpec,
    AnnotationSpec,
    CrossSourceComparison,
    DurationBounds,
    EntityReference,
    MediaReference,
    ReferenceKind,
    Severity,
    TemporalFields,
    TemporalOrigin,
    TimeUnit,
)
from conv_wm.data.datasets.ego4d_cleaning import build_annotation_cleaning
from conv_wm.data.datasets.ego4d_native_voice import load_ego4d_native_voice
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
        "valid": pa.Column(bool, nullable=False),
    },
    strict=True,
    name="ego4d_clips_clean",
)


EGO4D_MISSING_VOICE_SCHEMA = pa.DataFrameSchema(
    {
        "clip_uid": pa.Column(str, nullable=False),
        "person_id": pa.Column(str, nullable=True),
        "start_time": pa.Column(float, nullable=False),
        "end_time": pa.Column(float, nullable=False),
    },
    strict=True,
    name="ego4d_missing_voice_segments_clean",
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
        "start_frame": pa.Column(int, nullable=False),
        "end_frame": pa.Column(int, nullable=False),
        "video_start_time": pa.Column(float, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(int, nullable=False),
        "video_end_frame": pa.Column(int, nullable=False),
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
        "video_start_frame": pa.Column(int, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_end_frame": pa.Column(int, nullable=False),
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
        "target": pa.Column(str, nullable=True),
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
        "start_frame": pa.Column(int, nullable=False),
        "end_frame": pa.Column(int, nullable=False),
        "video_start_time": pa.Column(float, nullable=False),
        "video_end_time": pa.Column(float, nullable=False),
        "video_start_frame": pa.Column(int, nullable=False),
        "video_end_frame": pa.Column(int, nullable=False),
        "person": pa.Column(str, nullable=False),
        "target": pa.Column(str, nullable=True),
        "is_at_me": pa.Column(bool, nullable=False),
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
            ("missing_voice_segments_clean", EGO4D_MISSING_VOICE_SCHEMA),
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
        _clip_relation("missing_voice_segments_clean", "missing_voice_segments"),
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

# ---------------------------------------------------------------------------
# Annotation semantics
# ---------------------------------------------------------------------------

_CLIP_REF = EntityReference(
    kind=ReferenceKind.CLIP,
    columns=("clip_uid",),
    target_source="clips",
    target_columns=("clip_uid",),
)
_CLIP_SECONDS = TemporalFields(
    start="start_time",
    end="end_time",
    unit=TimeUnit.SECONDS,
    origin=TemporalOrigin.CLIP_START,
)
_CLIP_DURATION_BOUNDS = DurationBounds(
    source="clips",
    key_columns=("clip_uid",),
    target_key_columns=("clip_uid",),
    start_column="clip_start_sec",
    end_column="clip_end_sec",
)


def _person_ref(
    person_column: str, *unknown: str, severity: Severity = Severity.ERROR
) -> EntityReference:
    return EntityReference(
        kind=ReferenceKind.PARTICIPANT,
        columns=("clip_uid", person_column),
        target_source="persons",
        target_columns=("clip_uid", "person_id"),
        unknown_values=unknown,
        severity=severity,
    )


ANNOTATIONS = AnnotationSpec(
    sources=(
        AnnotationSourceSpec(
            dataset="ego4d",
            name="clips",
            description="Benchmark clips cut from full videos; the clip timeline starts at 0.",
            scope=AnnotationScope.CLIP,
            load=parquet_table("ego4d", "interim", "clips_clean"),
            dimensions=("clip_structure",),
            provenance=AnnotationProvenance.DETERMINISTIC_DERIVED,
            identity=("clip_uid",),
            temporal=TemporalFields(
                start="video_start_sec",
                end="video_end_sec",
                unit=TimeUnit.SECONDS,
                origin=TemporalOrigin.MEDIA_START,
                out_of_bounds_tolerance=1.0,
            ),
            media_reference=MediaReference(columns=("video_uid",)),
            bounds=DurationBounds(
                source="media",
                key_columns=("video_uid",),
                target_key_columns=("relative_path",),
            ),
            value_fields=("split", "clip_start_sec", "clip_end_sec"),
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="persons",
            description="Participants of a clip; person 0 is the camera wearer.",
            scope=AnnotationScope.PARTICIPANT,
            load=parquet_table("ego4d", "interim", "persons_clean"),
            dimensions=("participant",),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            identity=("clip_uid", "person_id"),
            entity_references=(_CLIP_REF,),
            value_fields=("is_camera_wearer",),
            known_limitations=(
                "The camera wearer has no face track and no looking annotation.",
            ),
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="voice_segments",
            description="Per-person voice activity intervals on the clip timeline.",
            scope=AnnotationScope.TEMPORAL_INTERVAL,
            load=parquet_table("ego4d", "interim", "voice_segments_clean"),
            dimensions=("speech",),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            temporal=_CLIP_SECONDS,
            entity_references=(_CLIP_REF, _person_ref("person_id")),
            bounds=_CLIP_DURATION_BOUNDS,
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="transcriptions",
            description="Transcribed utterances on the clip timeline, speaker may be unknown.",
            scope=AnnotationScope.TEMPORAL_INTERVAL,
            load=parquet_table("ego4d", "interim", "transcriptions_clean"),
            dimensions=("speech", "transcript"),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            temporal=TemporalFields(
                start="start_time_sec",
                end="end_time_sec",
                unit=TimeUnit.SECONDS,
                origin=TemporalOrigin.CLIP_START,
            ),
            entity_references=(_CLIP_REF, _person_ref("person_id", "-1")),
            bounds=_CLIP_DURATION_BOUNDS,
            value_fields=("transcription",),
            known_limitations=(
                (
                    "Transcriptions and voice segments are independent annotations; "
                    "join them by temporal overlap and person, never by equal timestamps."
                ),
                "Some clips have no transcription at all.",
            ),
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="social_segments_talking",
            description="Who talks to whom: speaker, target and whether the wearer is addressed.",
            scope=AnnotationScope.TEMPORAL_INTERVAL,
            load=parquet_table("ego4d", "interim", "social_segments_talking_clean"),
            dimensions=("speech", "addressee"),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            temporal=_CLIP_SECONDS,
            entity_references=(
                _CLIP_REF,
                _person_ref("person", "-1", severity=Severity.WARNING),
            ),
            bounds=_CLIP_DURATION_BOUNDS,
            value_fields=("target", "is_at_me"),
            known_limitations=(
                "`target` is release-defined nullable payload and is never imputed.",
                (
                    "Two retained rows name source person IDs absent from the clip's "
                    "persons collection; this is reported as a warning, not rewritten."
                ),
                (
                    "Addressee information is asymmetric with respect to the camera "
                    "wearer; population statistics, rather than an invented target, "
                    "describe unavailable explicit targets."
                ),
            ),
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="social_segments_looking",
            description="Looking-at-camera-wearer intervals from the social annotation.",
            scope=AnnotationScope.TEMPORAL_INTERVAL,
            load=parquet_table("ego4d", "interim", "social_segments_looking_clean"),
            dimensions=("gaze",),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            temporal=_CLIP_SECONDS,
            entity_references=(_CLIP_REF, _person_ref("person", "-1")),
            bounds=_CLIP_DURATION_BOUNDS,
            value_fields=("target", "is_at_me"),
            known_limitations=("`target` is nullable throughout the current release.",),
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="tracking_paths",
            description="Face tracks of a person within a clip (one row per track).",
            scope=AnnotationScope.PARTICIPANT,
            load=parquet_table("ego4d", "interim", "tracking_paths_clean"),
            dimensions=("person_tracking",),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            identity=("clip_uid", "person_id", "track_id"),
            entity_references=(_CLIP_REF, _person_ref("person_id")),
            value_fields=("unmapped_frames_count",),
        ),
        AnnotationSourceSpec(
            dataset="ego4d",
            name="tracks",
            description="Per-frame face boxes of each track, on the clip frame index.",
            scope=AnnotationScope.POINT_EVENT,
            load=parquet_table("ego4d", "interim", "tracks_clean"),
            dimensions=("person_tracking",),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            identity=("clip_uid", "person_id", "track_id", "clip_frame"),
            temporal=TemporalFields(
                start="clip_frame",
                end=None,
                unit=TimeUnit.FRAMES,
                origin=TemporalOrigin.CLIP_START,
            ),
            entity_references=(
                _CLIP_REF,
                EntityReference(
                    kind=ReferenceKind.PARTICIPANT,
                    columns=("clip_uid", "person_id", "track_id"),
                    target_source="tracking_paths",
                    target_columns=("clip_uid", "person_id", "track_id"),
                ),
            ),
            bounds=DurationBounds(
                source="clips",
                key_columns=("clip_uid",),
                target_key_columns=("clip_uid",),
                start_column="clip_start_frame",
                end_column="clip_end_frame",
            ),
            value_fields=("x", "y", "width", "height", "video_frame"),
        ),
    ),
    comparisons=(
        CrossSourceComparison(
            name="voice_segments_vs_transcriptions",
            left_source="voice_segments",
            right_source="transcriptions",
            left_key_columns=("clip_uid", "person_id"),
            right_key_columns=("clip_uid", "person_id"),
            description="Independent speech annotations; asymmetric coverage is expected.",
        ),
        CrossSourceComparison(
            name="voice_segments_vs_transcriptions_known_speakers",
            left_source="voice_segments",
            right_source="transcriptions",
            left_key_columns=("clip_uid", "person_id"),
            right_key_columns=("clip_uid", "person_id"),
            excluded_key_values=("-1",),
            description="Strict overlap after excluding the unknown-speaker sentinel.",
        ),
        CrossSourceComparison(
            name="voice_segments_vs_transcriptions_relaxed",
            left_source="voice_segments",
            right_source="transcriptions",
            left_key_columns=("clip_uid", "person_id"),
            right_key_columns=("clip_uid", "person_id"),
            overlap_tolerance=0.5,
            excluded_key_values=("-1",),
            description=(
                "Diagnostic ±0.5 s overlap expansion only; it does not alter source "
                "timestamps or define a canonical matching tolerance."
            ),
        ),
        CrossSourceComparison(
            name="voice_segments_vs_transcriptions_clip_level",
            left_source="voice_segments",
            right_source="transcriptions",
            left_key_columns=("clip_uid",),
            right_key_columns=("clip_uid",),
            description="Clip-level diagnostic that ignores speaker assignment.",
        ),
        CrossSourceComparison(
            name="voice_segments_vs_social_talking",
            left_source="voice_segments",
            right_source="social_segments_talking",
            left_key_columns=("clip_uid", "person_id"),
            right_key_columns=("clip_uid", "person"),
            description="Talking segments carry addressee labels for a subset of speech.",
        ),
        CrossSourceComparison(
            name="voice_segments_vs_social_talking_with_target",
            left_source="voice_segments",
            right_source="social_segments_talking",
            left_key_columns=("clip_uid", "person_id"),
            right_key_columns=("clip_uid", "person"),
            right_required_non_null=("target",),
            description="Same comparison restricted to rows with an explicit target.",
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
    annotations=ANNOTATIONS,
    annotation_cleaner=build_annotation_cleaning,
    native_focal_voice=load_ego4d_native_voice,
    audio=AudioInterpretation(
        known_boundary_grid=KnownBoundaryGrid(period_sec=STITCH_PERIOD_SEC),
        extra_decode_cases=systematic_profile_cases,
        summary_section=characterize_44100_regime,
    ),
)
