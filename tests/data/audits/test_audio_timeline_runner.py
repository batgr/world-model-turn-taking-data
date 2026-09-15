import pandas as pd

from conv_wm.data.audits.run_audio_timeline import (
    annotate_audio_events,
    build_dataset_sample_rate_summary,
    final_drift_threshold_flags,
)


def test_ego4d_pcm_event_is_annotated_on_observed_stitch_grid():
    events = pd.DataFrame(
        {
            "event_type": ["pcm_step_episode"],
            "event_time_sec": [300.00003125],
        }
    )

    result = annotate_audio_events(
        events,
        dataset="ego4d",
        sample_rate_hz=32000,
        reference_decoded_samples=1024,
        skip_samples=2048,
    )

    assert result.loc[0, "nearest_stitch_time_sec"] == 300.0
    assert bool(result.loc[0, "near_known_stitch_boundary"])
    assert result.loc[0, "event_category"] == ("stitch_related_pcm_discontinuity")


def test_early_event_is_kept_separate_from_stitch_interpretation():
    events = pd.DataFrame(
        {
            "event_type": ["pcm_step_episode"],
            "event_time_sec": [0.05],
        }
    )

    result = annotate_audio_events(
        events,
        dataset="egocom",
        sample_rate_hz=44100,
        reference_decoded_samples=1024,
        skip_samples=1024,
    )

    assert bool(result.loc[0, "has_early_boundary_context"])
    assert result.loc[0, "event_category"] == "codec_boundary_pcm_event"
    assert not bool(result.loc[0, "near_known_stitch_boundary"])


def test_long_episode_that_starts_early_is_not_a_codec_boundary_event():
    events = pd.DataFrame(
        {
            "event_type": ["pcm_step_episode"],
            "event_time_sec": [0.05],
            "event_end_time_sec": [60.0],
        }
    )

    result = annotate_audio_events(
        events,
        dataset="ego4d",
        sample_rate_hz=44100,
        reference_decoded_samples=1024,
        skip_samples=2048,
    )

    assert bool(result.loc[0, "has_early_boundary_context"])
    assert result.loc[0, "event_category"] == "pcm_timestamp_variation"


def test_audio_dropout_outside_priming_context_keeps_its_label():
    events = pd.DataFrame(
        {
            "event_type": ["audio_dropout"],
            "event_category": ["audio_dropout"],
            "event_time_sec": [300.0],
            "event_end_time_sec": [300.0],
        }
    )

    result = annotate_audio_events(
        events,
        dataset="ego4d",
        sample_rate_hz=32000,
        reference_decoded_samples=1024,
        skip_samples=2048,
    )

    # Exactly on the stitch grid, but a dropout is never relabelled as a stitch.
    assert bool(result.loc[0, "near_known_stitch_boundary"])
    assert result.loc[0, "event_category"] == "audio_dropout"


def test_persistent_step_in_priming_context_is_a_codec_boundary_event():
    events = pd.DataFrame(
        {
            "event_type": ["audio_dropout"],
            "event_category": ["audio_dropout"],
            "event_time_sec": [0.05],
            "event_end_time_sec": [0.05],
        }
    )

    result = annotate_audio_events(
        events,
        dataset="ego4d",
        sample_rate_hz=48000,
        reference_decoded_samples=1024,
        skip_samples=2048,
    )

    assert result.loc[0, "event_category"] == "codec_boundary_pcm_event"


def test_final_drift_thresholds_are_strict():
    exact = final_drift_threshold_flags(100.0, video_fps=30.0)

    assert not exact["final_pcm_drift_exceeds_100_ms"]
    assert final_drift_threshold_flags(100.0001, video_fps=30.0)[
        "final_pcm_drift_exceeds_100_ms"
    ]


def test_final_drift_thresholds_use_each_files_video_fps():
    below = final_drift_threshold_flags(10.0, video_fps=30.0)
    above = final_drift_threshold_flags(-20.0, video_fps=30.0)

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
            "n_abnormally_long_packets": [1],
            "dropout_duration_ms": [64.0],
            "dropout_total_excess_ms": [64.0],
        }
    )

    result = build_dataset_sample_rate_summary(files, events)["ego4d"]["48000"]

    assert result["final_pcm_drift_ms_quantiles"]["min"] == -20.0
    assert result["abs_final_pcm_drift_ms_quantiles"]["max"] == 600.0
    assert result["final_pcm_drift_threshold_counts"] == {
        "comparison": "strictly_greater",
        "above_1_ms": 2,
        "above_half_video_frame": 2,
        "above_100_ms": 1,
        "above_500_ms": 1,
    }
    dropouts = result["audio_dropouts"]
    assert dropouts["n_events"] == 1
    assert dropouts["n_persistent_steps"] == 1
    assert dropouts["n_candidate_steps"] == 41
    assert dropouts["n_compensated_candidate_steps"] == 40
    assert dropouts["n_files_with_compensated_cadence"] == 1
    # Long packets are an observation, not the dropout count.
    assert dropouts["n_abnormally_long_packets_observed"] == 41
