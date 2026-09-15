import math

import pandas as pd

from conv_wm.data.media.audio import (
    AudioTimelineEvent,
    EventCategory,
    EventDirection,
    EventType,
    KnownBoundaryGrid,
    final_drift_flags,
    interpret_events,
)
from conv_wm.data.media.audio.summary import by_dataset_and_sample_rate

STITCH_GRID = KnownBoundaryGrid(period_sec=300.0)


def _event(
    *,
    event_type: EventType = EventType.PCM_STEP_EPISODE,
    category: EventCategory = EventCategory.PCM_TIMESTAMP_VARIATION,
    time_sec: float,
    end_time_sec: float | None = None,
) -> AudioTimelineEvent:
    nan = math.nan
    return AudioTimelineEvent(
        packet_index=0,
        next_packet_index=1,
        event_type=event_type,
        event_category=category,
        event_direction=EventDirection.GAP,
        event_pts=0,
        event_time_sec=time_sec,
        event_end_time_sec=time_sec if end_time_sec is None else end_time_sec,
        n_steps=1,
        magnitude_samples=100.0,
        packet_timeline_error_samples=nan,
        first_pcm_step_error_samples=100.0,
        max_abs_pcm_step_error_samples=100.0,
        net_pcm_drift_samples=100.0,
        robust_nominal_packet_duration_samples=1024.0,
        long_packet_duration_threshold_samples=1536.0,
        n_abnormally_long_packets=0,
        max_packet_duration_samples=1024.0,
        n_candidate_steps=1,
        n_compensated_candidate_steps=0,
        dropout_duration_samples=nan,
        dropout_duration_ms=nan,
        dropout_total_excess_samples=nan,
        dropout_total_excess_ms=nan,
        dropout_min_residual_in_window_samples=nan,
        dropout_residual_at_window_end_samples=nan,
        dropout_window_truncated=False,
        reference_decoded_samples=1024,
    )


def _interpret(
    event: AudioTimelineEvent, *, grid, sample_rate_hz=32000, skip_samples=2048
):
    return interpret_events(
        [event],
        sample_rate_hz=sample_rate_hz,
        reference_decoded_samples=1024,
        skip_samples=skip_samples,
        boundary_grid=grid,
    )[0]


def test_pcm_event_on_known_boundary_grid_takes_the_grid_category():
    result = _interpret(_event(time_sec=300.00003125), grid=STITCH_GRID)

    assert result.nearest_known_boundary_time_sec == 300.0
    assert result.near_known_boundary
    assert result.event.event_category == EventCategory.STITCH_RELATED


def test_without_a_grid_nothing_is_relabelled_as_stitch():
    result = _interpret(_event(time_sec=300.0), grid=None)

    assert result.nearest_known_boundary_index is None
    assert not result.near_known_boundary
    assert result.event.event_category == EventCategory.PCM_TIMESTAMP_VARIATION


def test_early_event_is_a_codec_boundary_event():
    result = _interpret(
        _event(time_sec=0.05), grid=None, sample_rate_hz=44100, skip_samples=1024
    )

    assert result.has_early_boundary_context
    assert result.event.event_category == EventCategory.CODEC_BOUNDARY


def test_long_episode_that_starts_early_is_not_a_codec_boundary_event():
    result = _interpret(
        _event(time_sec=0.05, end_time_sec=60.0), grid=STITCH_GRID, sample_rate_hz=44100
    )

    assert result.has_early_boundary_context
    assert result.event.event_category == EventCategory.PCM_TIMESTAMP_VARIATION


def test_audio_dropout_on_the_grid_keeps_its_label():
    dropout = _event(
        event_type=EventType.AUDIO_DROPOUT,
        category=EventCategory.AUDIO_DROPOUT,
        time_sec=300.0,
    )

    result = _interpret(dropout, grid=STITCH_GRID)

    assert result.near_known_boundary
    assert result.event.event_category == EventCategory.AUDIO_DROPOUT


def test_persistent_step_in_priming_context_is_a_codec_boundary_event():
    dropout = _event(
        event_type=EventType.AUDIO_DROPOUT,
        category=EventCategory.AUDIO_DROPOUT,
        time_sec=0.05,
    )

    result = _interpret(dropout, grid=STITCH_GRID, sample_rate_hz=48000)

    assert result.event.event_category == EventCategory.CODEC_BOUNDARY


def test_final_drift_thresholds_are_strict():
    assert not final_drift_flags(100.0, video_fps=30.0)[
        "final_pcm_drift_exceeds_100_ms"
    ]
    assert final_drift_flags(100.0001, video_fps=30.0)["final_pcm_drift_exceeds_100_ms"]


def test_final_drift_thresholds_use_each_files_video_fps():
    below = final_drift_flags(10.0, video_fps=30.0)
    above = final_drift_flags(-20.0, video_fps=30.0)

    assert below["half_video_frame_ms"] == 500 / 30
    assert not below["final_pcm_drift_exceeds_half_video_frame"]
    assert above["final_pcm_drift_exceeds_half_video_frame"]


def test_dataset_rate_summary_reports_drift_quantiles_and_dropouts():
    files = pd.DataFrame(
        {
            "dataset": ["ego4d", "ego4d"],
            "sample_rate_hz": [48000, 48000],
            "probe_ok": [True, True],
            "final_cumulative_pcm_drift_ms": [-20.0, 600.0],
            "abs_final_cumulative_pcm_drift_ms": [20.0, 600.0],
            "final_pcm_drift_exceeds_1_ms": [True, True],
            "final_pcm_drift_exceeds_half_video_frame": [True, True],
            "final_pcm_drift_exceeds_100_ms": [False, True],
            "final_pcm_drift_exceeds_500_ms": [False, True],
            "has_audio_dropout": [True, False],
            "has_compensated_timestamp_cadence": [False, True],
            "n_dropout_candidate_steps": [1, 40],
            "n_compensated_candidate_steps": [0, 40],
            "n_abnormally_long_packet_durations": [1, 40],
        }
    )
    events = pd.DataFrame(
        {
            "dataset": ["ego4d"],
            "sample_rate_hz": [48000],
            "event_category": ["audio_dropout"],
            "n_steps": [1],
            "dropout_duration_ms": [64.0],
        }
    )

    result = by_dataset_and_sample_rate(
        files, events, threshold_comparison="strictly_greater"
    )
    section = result["ego4d"]["48000"]

    assert section["final_pcm_drift_ms_quantiles"]["min"] == -20.0
    assert section["abs_final_pcm_drift_ms_quantiles"]["max"] == 600.0
    assert section["final_pcm_drift_threshold_counts"] == {
        "comparison": "strictly_greater",
        "above_1_ms": 2,
        "above_half_video_frame": 2,
        "above_100_ms": 1,
        "above_500_ms": 1,
    }
    dropouts = section["audio_dropouts"]
    assert dropouts["n_events"] == 1
    assert dropouts["n_persistent_steps"] == 1
    assert dropouts["n_candidate_steps"] == 41
    assert dropouts["n_compensated_candidate_steps"] == 40
    assert dropouts["n_files_with_compensated_cadence"] == 1
    assert dropouts["n_abnormally_long_packets_observed"] == 41
    assert dropouts["dropout_duration_ms_quantiles"]["max"] == 64.0
