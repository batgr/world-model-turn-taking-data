"""EgoCom: dataset-specific knowledge (structural contracts and media notes)."""

from __future__ import annotations

import pandera.pandas as pa
from pandera import Check

from conv_wm.data.annotations import (
    AnnotationProvenance,
    AnnotationScope,
    AnnotationSourceSpec,
    AnnotationSpec,
    DurationBounds,
    EntityReference,
    MediaReference,
    ReferenceKind,
    Severity,
    TemporalFields,
    TemporalOrigin,
    TimeUnit,
)
from conv_wm.data.datasets.spec import (
    DatasetSpec,
    RelationSpec,
    StructuralSpec,
    TableSpec,
    csv_table,
)

EGOCOM_VIDEO_INFO_SCHEMA = pa.DataFrameSchema(
    {
        "video_id": pa.Column(int, nullable=False, unique=True),
        "conversation_id": pa.Column(str, nullable=False),
        "video_speaker_id": pa.Column(int, nullable=False),
        "num_speakers": pa.Column(int, nullable=False, checks=Check.gt(0)),
        "speaker_name": pa.Column(str, nullable=True),
        "speaker_gender": pa.Column(str, nullable=True),
        "duration_seconds": pa.Column(int, nullable=False, checks=Check.gt(0)),
        "word_count": pa.Column(int, nullable=True),
        "speaker_is_host": pa.Column(bool, nullable=True),
        "tokenized_words": pa.Column(str, nullable=True),
        "native_speaker": pa.Column(bool, nullable=True),
        "video_name": pa.Column(str, nullable=False),
        "background_fan": pa.Column(bool, nullable=True),
        "background_music": pa.Column(bool, nullable=True),
        "cid": pa.Column(str, nullable=True),
        "train": pa.Column(bool, nullable=False),
        "val": pa.Column(bool, nullable=False),
        "test": pa.Column(bool, nullable=False),
    },
    checks=Check(
        lambda df: df[["train", "val", "test"]].sum(axis=1).eq(1),
        error="Exactly one of train, val, and test must be true.",
    ),
    strict=True,
    name="egocom_video_info",
)


EGOCOM_GROUND_TRUTH_SCHEMA = pa.DataFrameSchema(
    {
        "conversation_id": pa.Column(str, nullable=False),
        "endTime": pa.Column(float, nullable=True),
        "speaker_id": pa.Column(int, nullable=False),
        "startTime": pa.Column(float, nullable=True),
        "word": pa.Column(str, nullable=True),
    },
    strict=True,
    name="egocom_ground_truth_transcriptions",
)


STRUCTURE = StructuralSpec(
    tables=(
        TableSpec(
            name="video_info",
            schema=EGOCOM_VIDEO_INFO_SCHEMA,
            load=csv_table("egocom", "raw", "video_info"),
            description="One row per participant recording (5-minute part).",
        ),
        TableSpec(
            name="ground_truth_transcriptions",
            schema=EGOCOM_GROUND_TRUTH_SCHEMA,
            load=csv_table("egocom", "raw", "ground_truth"),
            description="Word-level transcript with per-word timing per conversation part.",
        ),
    ),
    relations=(
        RelationSpec(
            name="ground_truth_conversation_id_in_video_info",
            child_table="ground_truth_transcriptions",
            child_columns=("conversation_id",),
            parent_table="video_info",
            parent_columns=("conversation_id",),
        ),
    ),
)

ANNOTATIONS = AnnotationSpec(
    sources=(
        AnnotationSourceSpec(
            dataset="egocom",
            name="video_info",
            description=(
                "One participant's recording of one conversation part, with speaker traits "
                "and recording conditions."
            ),
            scope=AnnotationScope.PARTICIPANT_INTERACTION,
            load=csv_table("egocom", "raw", "video_info"),
            dimensions=(
                "participant_traits",
                "recording_conditions",
                "interaction_structure",
            ),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            identity=("video_id",),
            media_reference=MediaReference(columns=("video_name",)),
            value_fields=(
                "conversation_id",
                "cid",
                "video_speaker_id",
                "num_speakers",
                "speaker_gender",
                "speaker_is_host",
                "native_speaker",
                "background_fan",
                "background_music",
                "duration_seconds",
            ),
            known_limitations=(
                "`duration_seconds` is an integer declared duration, not the measured stream length.",
            ),
        ),
        AnnotationSourceSpec(
            dataset="egocom",
            name="ground_truth_transcriptions",
            description="Word-level transcript per conversation part; punctuation tokens are untimed.",
            scope=AnnotationScope.TEMPORAL_INTERVAL,
            load=csv_table("egocom", "raw", "ground_truth"),
            dimensions=("speech", "transcript"),
            provenance=AnnotationProvenance.HUMAN_OBSERVED,
            temporal=TemporalFields(
                start="startTime",
                end="endTime",
                unit=TimeUnit.SECONDS,
                origin=TemporalOrigin.INTERACTION_START,
                nullable=True,
            ),
            entity_references=(
                EntityReference(
                    kind=ReferenceKind.INTERACTION,
                    columns=("conversation_id",),
                    target_source="video_info",
                    target_columns=("conversation_id",),
                ),
                # Recorded participants are the only participant table EgoCom ships;
                # a speaker without a recording is a documented caveat, not a defect.
                EntityReference(
                    kind=ReferenceKind.PARTICIPANT,
                    columns=("conversation_id", "speaker_id"),
                    target_source="video_info",
                    target_columns=("conversation_id", "video_speaker_id"),
                    severity=Severity.WARNING,
                ),
            ),
            bounds=DurationBounds(
                source="video_info",
                key_columns=("conversation_id",),
                target_key_columns=("conversation_id",),
                duration_column="duration_seconds",
            ),
            value_fields=("word",),
            known_limitations=(
                "About half of the rows are untimed tokens (punctuation, empty words).",
                "Timestamps are relative to the conversation part, not to the 20-minute video.",
                (
                    "Seven two-speaker conversations (13 parts, 13,512 rows, 9,007 non-empty "
                    "words) attribute speech to speaker 3, who has neither a recording on "
                    "disk nor a participant record; the identity of that speaker is unresolved."
                ),
            ),
        ),
    ),
)

EGOCOM = DatasetSpec(
    name="egocom",
    description="EgoCom multi-person egocentric conversations (240p, 20-minute videos).",
    structure=STRUCTURE,
    annotations=ANNOTATIONS,
)
