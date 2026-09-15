import shutil
import subprocess

import pandas as pd
import pytest

from conv_wm.data.media.video_timeline import (
    analyze_frame_timeline,
    known_boundaries_within,
    make_timeline_windows,
    probe_video_timestamps,
)

FPS = 30.0
TIME_BASE = "1/15360"


def _cfr_frames(n: int) -> pd.DataFrame:
    return pd.DataFrame({"pts_time": [i / FPS for i in range(n)]})


def test_analyze_frame_timeline_empty_frames_is_insufficient_data():
    result = analyze_frame_timeline(
        pd.DataFrame(),
        fps=FPS,
        time_base=TIME_BASE,
    )

    assert result.to_row()["timeline_class"] == "insufficient_data"
    assert result.to_row()["n_frames"] == 0
    assert result.to_row()["timestamp_coverage"] == 0.0


def test_analyze_frame_timeline_cfr_is_consistent():
    result = analyze_frame_timeline(
        _cfr_frames(300),
        fps=FPS,
        time_base=TIME_BASE,
    )

    assert result.to_row()["timeline_class"] == "cfr_consistent"
    assert result.to_row()["monotonic"] is True
    assert result.to_row()["n_duplicate_timestamps"] == 0
    assert result.to_row()["max_delta_error_ticks"] <= 1.0


def test_analyze_frame_timeline_detects_dropped_frame():
    frames = _cfr_frames(300).drop(index=150).reset_index(drop=True)

    result = analyze_frame_timeline(
        frames,
        fps=FPS,
        time_base=TIME_BASE,
    )

    assert result.to_row()["timeline_class"] == "vfr_or_irregular"
    assert result.to_row()["delta_max_sec"] == pytest.approx(2 / FPS)


def test_analyze_frame_timeline_ignores_untimestamped_packets():
    frames = _cfr_frames(10)
    frames.loc[len(frames)] = {"pts_time": None}

    result = analyze_frame_timeline(
        frames,
        fps=FPS,
        time_base=TIME_BASE,
    )

    assert result.to_row()["n_frames"] == 11
    assert result.to_row()["n_timestamped"] == 10
    assert result.to_row()["timeline_class"] == "cfr_consistent"


def test_make_timeline_windows_short_video_is_single_window():
    assert make_timeline_windows(5.0, window_sec=10.0) == {"full_short": "%+5.000000"}


def test_make_timeline_windows_long_video_has_three_windows():
    windows = make_timeline_windows(100.0, window_sec=10.0)

    assert windows == {
        "start": "%+10.000000",
        "middle": "45.000000%+10.000000",
        "end": "90.000000%+10.000000",
    }


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")

    path = tmp_path_factory.mktemp("media") / "cfr.mp4"

    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )

    return path


def test_probe_video_timestamps_returns_sorted_presentation_order(synthetic_video):
    packets = probe_video_timestamps(synthetic_video)

    assert len(packets) == 60
    assert packets["pts_time"].is_monotonic_increasing

    result = analyze_frame_timeline(
        packets,
        fps=FPS,
        time_base=TIME_BASE,
    )

    assert result.to_row()["timeline_class"] == "cfr_consistent"


def test_probe_video_timestamps_honours_read_intervals(synthetic_video):
    packets = probe_video_timestamps(
        synthetic_video,
        read_intervals="%+1",
    )

    assert 0 < len(packets) < 60


def test_probe_video_timestamps_raises_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        probe_video_timestamps(tmp_path / "missing.mp4")


def test_known_boundaries_add_windows_inside_the_stream():
    windows = make_timeline_windows(
        1000.0, window_sec=10.0, known_boundaries_sec=(300.0, 600.0, 900.0, 1200.0)
    )

    assert set(windows) == {
        "start",
        "middle",
        "end",
        "boundary_1",
        "boundary_2",
        "boundary_3",
    }
    assert windows["boundary_1"] == "295.000000%+10.000000"


def test_known_boundaries_within_excludes_the_stream_end():
    assert known_boundaries_within(900.0, period_sec=300.0) == (300.0, 600.0)
    assert known_boundaries_within(1000.0, period_sec=None) == ()
