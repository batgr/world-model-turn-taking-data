"""Pandera structural contracts for the project tables."""

import pandera.pandas as pa
from pandera import Check

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


EGO4D_SCHEMAS = {
    "clips_clean": EGO4D_CLIPS_SCHEMA,
    "persons_clean": EGO4D_PERSONS_SCHEMA,
    "tracking_paths_clean": EGO4D_TRACKING_PATHS_SCHEMA,
    "tracks_clean": EGO4D_TRACKS_SCHEMA,
    "voice_segments_clean": EGO4D_VOICE_SEGMENTS_SCHEMA,
    "transcriptions_clean": EGO4D_TRANSCRIPTIONS_SCHEMA,
    "social_segments_talking_clean": EGO4D_SOCIAL_SEGMENTS_TALKING_SCHEMA,
    "social_segments_looking_clean": EGO4D_SOCIAL_SEGMENTS_LOOKING_SCHEMA,
}
