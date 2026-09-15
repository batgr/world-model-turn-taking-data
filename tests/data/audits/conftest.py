import pandas as pd
import pytest


@pytest.fixture
def valid_egocom_tables():
    video_info = pd.DataFrame(
        {
            "video_id": [1, 2],
            "conversation_id": ["c1", "c2"],
            "video_speaker_id": [1, 2],
            "num_speakers": [2, 2],
            "speaker_name": ["alice", "bob"],
            "speaker_gender": ["female", "male"],
            "duration_seconds": [120, 150],
            "word_count": [100, 120],
            "speaker_is_host": [True, False],
            "tokenized_words": ["hello", "hi"],
            "native_speaker": [True, True],
            "video_name": ["v1", "v2"],
            "background_fan": [False, False],
            "background_music": [False, False],
            "cid": ["c1", "c2"],
            "train": [True, False],
            "val": [False, True],
            "test": [False, False],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "conversation_id": ["c1", "c2"],
            "endTime": [1.0, None],
            "speaker_id": [1, 2],
            "startTime": [0.5, None],
            "word": ["hello", None],
        }
    )
    return video_info, ground_truth


@pytest.fixture
def valid_ego4d_tables():
    return {
        "clips_clean": pd.DataFrame(
            {
                "split": ["train"],
                "clip_uid": ["clip-1"],
                "source_clip_uid": ["source-1"],
                "video_uid": ["video-1"],
                "video_start_sec": [10.0],
                "video_end_sec": [20.0],
                "video_start_frame": [300],
                "video_end_frame": [600],
                "clip_start_sec": [0],
                "clip_end_sec": [10.0],
                "clip_start_frame": [0],
                "clip_end_frame": [300],
            }
        ),
        "persons_clean": pd.DataFrame(
            {
                "clip_uid": ["clip-1"],
                "person_id": ["0"],
                "is_camera_wearer": [True],
            }
        ),
        "tracking_paths_clean": pd.DataFrame(
            {
                "clip_uid": ["clip-1"],
                "person_id": ["0"],
                "track_id": ["track-1"],
                "unmapped_frames_count": [0],
                "unmapped_frames": [object()],
            }
        ),
        "tracks_clean": pd.DataFrame(
            {
                "clip_uid": ["clip-1"],
                "person_id": ["0"],
                "track_id": ["track-1"],
                "x": [1.0],
                "y": [2.0],
                "width": [3.0],
                "height": [4.0],
                "clip_frame": [0],
                "video_frame": [300],
            }
        ),
        "voice_segments_clean": pd.DataFrame(
            {
                "clip_uid": ["clip-1"],
                "start_time": [0.0],
                "end_time": [1.0],
                "start_frame": [0],
                "end_frame": [30],
                "video_start_time": [10.0],
                "video_end_time": [11.0],
                "video_start_frame": [300],
                "video_end_frame": [330],
                "person_id": ["0"],
            }
        ),
        "transcriptions_clean": pd.DataFrame(
            {
                "clip_uid": ["clip-1"],
                "transcription": ["hello"],
                "start_time_sec": [0.0],
                "end_time_sec": [1.0],
                "person_id": ["-1"],
                "video_start_time": [10.0],
                "video_start_frame": [300],
                "video_end_time": [11.0],
                "video_end_frame": [330],
            }
        ),
        "social_segments_talking_clean": pd.DataFrame(
            {
                "clip_uid": ["clip-1"],
                "start_time": [0.0],
                "end_time": [1.0],
                "start_frame": [0],
                "end_frame": [30],
                "video_start_time": [10.0],
                "video_end_time": [11.0],
                "video_start_frame": [300],
                "video_end_frame": [330],
                "person": ["-1"],
                "target": ["0"],
                "is_at_me": [True],
            }
        ),
        "social_segments_looking_clean": pd.DataFrame(
            {
                "clip_uid": pd.Series(dtype="str"),
                "start_time": pd.Series(dtype="float64"),
                "end_time": pd.Series(dtype="float64"),
                "start_frame": pd.Series(dtype="int64"),
                "end_frame": pd.Series(dtype="int64"),
                "video_start_time": pd.Series(dtype="float64"),
                "video_end_time": pd.Series(dtype="float64"),
                "video_start_frame": pd.Series(dtype="int64"),
                "video_end_frame": pd.Series(dtype="int64"),
                "person": pd.Series(dtype="str"),
                "target": pd.Series(dtype="str"),
                "is_at_me": pd.Series(dtype="bool"),
            }
        ),
    }
