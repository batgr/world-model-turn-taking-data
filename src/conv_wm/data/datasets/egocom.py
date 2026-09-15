"""EgoCom: dataset-specific knowledge (structural contracts and media notes)."""

from __future__ import annotations

import pandera.pandas as pa
from pandera import Check

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

EGOCOM = DatasetSpec(
    name="egocom",
    description="EgoCom multi-person egocentric conversations (240p, 20-minute videos).",
    structure=STRUCTURE,
)
