import pandas as pd
import pytest

from conv_wm.data.media.audio import (
    AudioFileTimelineRecord,
    AudioTimelineParameters,
    TimelineStatus,
    analyze_audio_packets,
    decoded_frame_continuity,
    effective_audio_bounds,
    extract_skip_samples,
    summarize_decoded_sample_counts,
)


def analyze_audio_packet_timeline(
    packets: list[dict],
    *,
    sample_rate_hz: int,
    time_base: str,
    reference_decoded_samples: int,
    **overrides: float,
) -> tuple[dict, pd.DataFrame]:
    """Run the typed analysis and return (flat file row, events frame)."""
    parameters = AudioTimelineParameters(
        reference_decoded_samples=reference_decoded_samples, **overrides
    )
    file = AudioFileTimelineRecord(
        dataset="test",
        relative_path="test.mp4",
        audio_codec="aac",
        sample_rate_hz=sample_rate_hz,
        time_base=time_base,
        audio_duration_sec=0.0,
        video_avg_frame_rate=30.0,
        reference_decoded_samples_source="test",
        probe_ok=True,
        probe_error=None,
        timeline_status=TimelineStatus.MEASURED,
        parameters=parameters,
    )
    result = analyze_audio_packets(packets, file=file)
    return result.file.to_row(), result.events_frame()


def test_empty_audio_timeline_is_insufficient():
    result = decoded_frame_continuity(
        pd.DataFrame(),
        sample_rate_hz=48000,
        time_base="1/48000",
    )

    assert result["timeline_class"] == "insufficient_data"
    assert result["n_frames"] == 0


def test_empty_packet_timeline_is_insufficient():
    summary, events = analyze_audio_packet_timeline(
        [],
        sample_rate_hz=48000,
        time_base="1/48000",
        reference_decoded_samples=1024,
    )

    assert summary["timeline_status"] == "insufficient_data"
    assert summary["n_packets"] == 0
    assert summary["timestamp_coverage"] == 0.0
    assert events.empty


def test_continuous_audio_timeline():
    frames = pd.DataFrame(
        {
            "pts": [
                0,
                1024,
                2048,
            ],
            "nb_samples": [
                1024,
                1024,
                1024,
            ],
        }
    )

    result = decoded_frame_continuity(
        frames,
        sample_rate_hz=48000,
        time_base="1/48000",
    )

    assert result["timeline_class"] == "continuous"
    assert result["n_gaps"] == 0
    assert result["n_overlaps"] == 0


def test_audio_gap_is_detected():
    frames = pd.DataFrame(
        {
            "pts": [
                0,
                1024,
                2050,
            ],
            "nb_samples": [
                1024,
                1024,
                1024,
            ],
        }
    )

    result = decoded_frame_continuity(
        frames,
        sample_rate_hz=48000,
        time_base="1/48000",
    )

    assert result["timeline_class"] == "gap_or_overlap"
    assert result["n_gaps"] == 1
    assert result["n_overlaps"] == 0
    assert result["max_gap_samples"] == pytest.approx(2.0)


def test_audio_overlap_is_detected():
    frames = pd.DataFrame(
        {
            "pts": [
                0,
                1024,
                2046,
            ],
            "nb_samples": [
                1024,
                1024,
                1024,
            ],
        }
    )

    result = decoded_frame_continuity(
        frames,
        sample_rate_hz=48000,
        time_base="1/48000",
    )

    assert result["n_gaps"] == 0
    assert result["n_overlaps"] == 1
    assert result["min_gap_samples"] == pytest.approx(-2.0)


def test_extract_skip_samples():
    packets = [
        {
            "pts": -2048,
            "pts_time": "-0.064",
            "side_data_list": [
                {
                    "side_data_type": ("Skip Samples"),
                    "skip_samples": 2048,
                    "discard_padding": 128,
                    "skip_reason": 0,
                    "discard_reason": 0,
                }
            ],
        }
    ]

    result = extract_skip_samples(packets)

    assert len(result) == 1
    assert result.iloc[0]["skip_samples"] == 2048
    assert result.iloc[0]["discard_padding"] == 128


def test_effective_audio_bounds():
    frames = pd.DataFrame(
        {
            "pts": [
                0,
                1024,
            ],
            "nb_samples": [
                1024,
                1024,
            ],
        }
    )

    start, end = effective_audio_bounds(
        frames,
        sample_rate_hz=48000,
        time_base="1/48000",
    )

    assert start == pytest.approx(0.0)

    assert end == pytest.approx(2048 / 48000)


def test_packet_jitter_without_persistent_pcm_drift():
    packets = [
        {
            "pts": 0,
            "duration": 1021,
        },
        {
            "pts": 1021,
            "duration": 1027,
        },
        {
            "pts": 2048,
            "duration": 1024,
        },
        {
            "pts": 3072,
            "duration": 1024,
        },
    ]

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=32000,
        time_base="1/32000",
        reference_decoded_samples=1024,
    )

    # Container timeline itself is perfect.
    assert summary["packet_timeline_gap_count"] == 0
    assert summary["packet_timeline_overlap_count"] == 0

    # PCM-vs-PTS clock briefly moves -3 samples,
    # then recovers.
    assert summary["final_cumulative_pcm_drift_samples"] == pytest.approx(0.0)

    assert summary["max_abs_cumulative_pcm_drift_samples"] == pytest.approx(3.0)

    assert summary["max_abs_pcm_step_error_samples"] == pytest.approx(3.0)

    assert summary["has_small_bounded_timestamp_jitter"]
    assert not summary["has_significant_final_pcm_offset"]

    assert events.empty


def test_persistent_pcm_clock_offset_is_separate_from_packet_clock():
    packets = [
        {
            "pts": 0,
            "duration": 1,
        },
        {
            "pts": 1,
            "duration": 1023,
        },
        {
            "pts": 1024,
            "duration": 1024,
        },
    ]

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=32000,
        time_base="1/32000",
        reference_decoded_samples=1024,
    )

    assert summary["packet_timeline_gap_count"] == 0

    assert summary["packet_timeline_overlap_count"] == 0

    assert summary["final_cumulative_pcm_drift_samples"] == pytest.approx(-1024.0)

    assert summary["max_abs_pcm_step_error_samples"] == pytest.approx(1023.0)

    assert summary["has_significant_final_pcm_offset"]

    event = events.loc[events["event_type"] == "pcm_step_episode"].iloc[0]
    assert event["net_pcm_drift_samples"] == pytest.approx(-1024.0)
    assert "pcm_cumulative_offset_peak" not in set(events["event_type"])
    assert summary["max_abs_cumulative_pcm_drift_samples"] == 1024.0
    assert summary["max_abs_cumulative_pcm_drift_time_sec"] == pytest.approx(
        1024 / 32000
    )


def _packets_from_durations(durations: list[int]) -> list[dict[str, int]]:
    packets, pts = [], 0
    for duration in durations:
        packets.append({"pts": pts, "duration": duration})
        pts += duration
    return packets


def test_persistent_long_packet_is_an_audio_dropout():
    packets = _packets_from_durations([1024, 4096] + [1024] * 200)

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=48000,
        time_base="1/48000",
        reference_decoded_samples=1024,
    )

    # Long duration is recorded as an observation...
    assert summary["n_abnormally_long_packet_durations"] == 1
    # ...and the category comes from persistence of the excess, not from length.
    assert summary["n_dropout_candidate_steps"] == 1
    assert summary["n_persistent_dropout_steps"] == 1
    assert summary["n_compensated_candidate_steps"] == 0

    assert list(events["event_category"]) == ["audio_dropout"]
    dropout = events.iloc[0]
    assert dropout["event_type"] == "audio_dropout"
    assert dropout["n_steps"] == 1
    assert dropout["dropout_duration_samples"] == 3072.0
    assert dropout["dropout_duration_ms"] == pytest.approx(64.0)
    assert dropout["dropout_min_residual_in_window_samples"] == 3072.0
    assert not dropout["dropout_window_truncated"]
    assert "cumulative_pcm_drift_samples" not in events.columns


def test_dense_compensating_cadence_is_not_a_dropout():
    # Nine short steps then one long step: locally compensating, zero net drift.
    cycle = [885] * 9 + [10 * 1024 - 9 * 885]
    packets = _packets_from_durations(cycle * 50)

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=44100,
        time_base="1/44100",
        reference_decoded_samples=1024,
    )

    assert summary["n_abnormally_long_packet_durations"] == 50
    assert summary["n_dropout_candidate_steps"] == 49
    assert summary["n_persistent_dropout_steps"] == 0
    assert summary["n_compensated_candidate_steps"] == 49
    assert summary["max_abs_cumulative_pcm_drift_samples"] == 9 * (1024 - 885)

    assert list(events["event_category"]) == ["compensated_timestamp_cadence"]
    episode = events.iloc[0]
    assert episode["n_candidate_steps"] == 49
    assert episode["n_compensated_candidate_steps"] == 49
    assert episode["n_abnormally_long_packets"] == 49
    assert pd.isna(episode["dropout_duration_samples"])


def test_slow_compensation_beyond_window_is_still_a_dropout():
    # +900-sample jump, then -5 samples every 170 packets: the excess survives the
    # two-second compensation window even though the file eventually catches up.
    packets = _packets_from_durations(
        [1024] * 100 + [1924] + ([1024] * 169 + [1019]) * 30
    )

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=48000,
        time_base="1/48000",
        reference_decoded_samples=1024,
        compensation_window_sec=2.0,
    )

    assert summary["compensation_window_steps"] == 94
    assert summary["n_persistent_dropout_steps"] == 1
    dropout = events.loc[events["event_category"] == "audio_dropout"].iloc[0]
    assert dropout["dropout_duration_samples"] == 900.0
    assert dropout["dropout_min_residual_in_window_samples"] == 900.0
    # The slow -5 steps (0.1 ms each) stay below the threshold: no other event.
    assert list(events["event_category"]) == ["audio_dropout"]


def test_dropout_splits_surrounding_episode():
    # Significant jitter, a persistent dropout, jitter again: three events.
    packets = _packets_from_durations([1024, 1124, 924, 4096, 1124, 924] + [1024] * 100)

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=48000,
        time_base="1/48000",
        reference_decoded_samples=1024,
    )

    # +100/-100 pairs are compensated candidates; the 4096 packet is not.
    assert list(events["event_category"]) == [
        "compensated_timestamp_cadence",
        "audio_dropout",
        "compensated_timestamp_cadence",
    ]
    assert list(events["n_steps"]) == [2, 1, 2]
    # The peak (+3072 then +100) sits one packet after the dropout itself.
    assert summary["max_abs_cumulative_pcm_drift_samples"] == 3172.0
    assert summary["max_abs_cumulative_pcm_drift_time_sec"] == pytest.approx(
        events.iloc[1]["event_time_sec"] + 1124 / 48000
    )


def test_initial_pcm_clock_gap():
    packets = [
        {
            "pts": 0,
            "duration": 1100,
        },
        {
            "pts": 1100,
            "duration": 1024,
        },
        {
            "pts": 2124,
            "duration": 1024,
        },
    ]

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=32000,
        time_base="1/32000",
        reference_decoded_samples=1024,
    )

    assert summary["packet_timeline_gap_count"] == 0

    assert summary["max_abs_pcm_step_error_samples"] == pytest.approx(76.0)

    assert summary["final_cumulative_pcm_drift_samples"] == pytest.approx(76.0)

    # A persistent positive step is a dropout at analyzer level; the runner
    # relabels it as a codec-boundary event when it sits in the priming context.
    assert list(events["event_type"]) == ["audio_dropout"]


def test_rational_time_base_conversion_uses_integer_pts():
    packets = [
        {"pts": 0, "duration": 1920},
        {"pts": 1920, "duration": 1920},
        {"pts": 3840, "duration": 1920},
    ]

    summary, events = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=48000,
        time_base="1/90000",
        reference_decoded_samples=1024,
    )

    assert summary["samples_per_time_base_tick_numerator"] == 8
    assert summary["samples_per_time_base_tick_denominator"] == 15
    assert summary["final_cumulative_pcm_drift_samples"] == pytest.approx(0.0)
    assert events.empty


def test_missing_packet_fields_reduce_coverage_without_bridging_steps():
    packets = [
        {"pts": 0, "duration": 1024},
        {"pts": 1024},
        {"duration": 1024},
        {"pts": 3072, "duration": 1024},
    ]

    summary, _ = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=48000,
        time_base="1/48000",
        reference_decoded_samples=1024,
    )

    assert summary["timestamp_coverage"] == pytest.approx(0.75)
    assert summary["duration_coverage"] == pytest.approx(0.75)
    assert summary["n_valid_pts_steps"] == 1
    assert summary["n_valid_packet_timeline_steps"] == 1


def test_last_packet_duration_does_not_create_a_false_timeline_gap():
    packets = [
        {"pts": 0, "duration": 1024},
        {"pts": 1024, "duration": 17},
    ]

    summary, _ = analyze_audio_packet_timeline(
        packets,
        sample_rate_hz=48000,
        time_base="1/48000",
        reference_decoded_samples=1024,
    )

    assert summary["packet_timeline_gap_count"] == 0
    assert summary["final_cumulative_pcm_drift_samples"] == pytest.approx(0.0)
    assert summary["max_abs_packet_duration_deviation_samples"] == 1007.0


def test_decoded_sample_reference_uses_mode_and_reports_960_frames():
    frames = pd.DataFrame({"nb_samples": [1024, 1024, 1024, 960, None]})

    result = summarize_decoded_sample_counts(frames)

    assert result.reference_decoded_samples == 1024
    assert result.distribution == {"960": 1, "1024": 3}
    assert result.n_non_reference_decoded_frames == 1
    assert result.n_960_sample_frames == 1
