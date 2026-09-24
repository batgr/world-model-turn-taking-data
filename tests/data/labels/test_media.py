"""Media extractors: numeric features on synthetic signals, masks, optional tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from label_helpers import recording

from conv_wm.data.labels import media
from conv_wm.data.labels.registry import Extractor, Table
from conv_wm.data.labels.speech import GridKeys
from conv_wm.data.labels.timeline import LabelConfig, derive

RATE = 16_000
STEP = 0.1


def sine(freq: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_audio_cell_features_of_a_sine():
    features = media.audio_cell_features(sine(440.0, 1.0), RATE, cells=12, step_s=STEP)
    assert features["audio_valid"].tolist() == [True] * 10 + [False] * 2
    assert features["global_audio_rms"][:10] == pytest.approx(
        0.5 / np.sqrt(2), rel=1e-3
    )
    assert features["spectral_centroid"][:10] == pytest.approx(440.0, rel=0.05)
    assert features["zero_crossing_rate"][:10] == pytest.approx(
        2 * 440 / RATE, rel=0.05
    )
    assert np.isnan(features["spectral_flux"][0])
    assert features["spectral_flux"][1:10] == pytest.approx(0.0, abs=1e-3)
    assert np.isnan(features["global_audio_rms"][10:]).all()


def test_digital_silence_has_no_spectral_features():
    features = media.audio_cell_features(np.zeros(RATE, np.float32), RATE, 10, STEP)
    assert not features["spectral_valid"].any()
    assert np.isnan(features["spectral_centroid"]).all()
    assert features["global_audio_rms"] == pytest.approx(0.0)


def overlap_structure():
    facts = recording(
        {"w": [(1.0, 2.0)], "x": [(1.5, 2.5)]}, end=3.0, unknown=[(2.8, 3.0)]
    )
    return derive(facts, LabelConfig())


def test_the_ego_solo_mask_excludes_overlap_and_unknown():
    structure = overlap_structure()
    starts = np.arange(30) * STEP
    fraction = media.ego_solo_fraction(structure, starts, STEP)
    assert fraction[10:15] == pytest.approx(1.0)
    assert fraction[15:20] == pytest.approx(
        0.0
    )  # overlap: never attributed to the wearer
    assert np.isnan(fraction[28:30]).all()  # UNKNOWN time
    times = np.asarray([1.2, 1.7, 2.2])
    assert media.ego_solo_mask(structure, times).tolist() == [True, False, False]


def test_ego_rms_uses_only_masked_samples():
    structure = overlap_structure()
    signal = np.concatenate([sine(200, 1.5, 0.1), sine(200, 1.5, 0.9)])
    starts = np.arange(30) * STEP
    fraction = media.ego_solo_fraction(structure, starts, STEP)
    mask = media.ego_solo_mask(structure, np.arange(len(signal)) / RATE)
    db = media.ego_rms_db(signal, RATE, mask, fraction, 30, STEP, 0.5)
    assert db[12] == pytest.approx(20 * np.log10(0.1 / np.sqrt(2)), abs=0.1)
    assert np.isnan(db[16])  # overlap
    assert np.isnan(db[5])  # nobody speaks


def test_pitch_cells_take_the_median_of_voiced_masked_frames():
    times = np.arange(0.0, 0.3, 0.01)
    f0 = np.full(len(times), 120.0)
    f0[::3] = np.nan
    mask = times < 0.2
    fraction = np.asarray([1.0, 1.0, 0.0])
    median, voiced = media.pitch_cells(
        times, f0, mask, fraction, np.asarray([0.0, 0.1, 0.2]), STEP, 0.5
    )
    assert median[:2] == pytest.approx([120.0, 120.0])
    assert voiced[0] == pytest.approx(0.6, abs=0.11)
    assert np.isnan(median[2]) and np.isnan(voiced[2])


def test_frame_features_of_uniform_red_frames():
    red = np.zeros((20, 30, 3), np.uint8)
    red[..., 0] = 255
    first = media.frame_features(red, None)
    second = media.frame_features(red, red)
    assert first[:3] == pytest.approx([255.0, 0.0, 0.0])
    assert first[5] == pytest.approx(1.0)  # saturation
    assert first[6] == pytest.approx(0.0)  # no edges: blur score 0
    assert first[7:10] == pytest.approx([224.0, 32.0, 32.0])
    assert np.isnan(first[10]) and second[10] == pytest.approx(0.0)


def test_video_cells_average_their_subframes_and_flag_missing_frames():
    frames = np.stack(
        [media.frame_features(np.full((8, 8, 3), v, np.uint8), None) for v in range(7)]
    )
    cells = media.video_cell_features(frames, cells=3, subframes=3)
    assert cells["frame_valid"].tolist() == [True, True, False]
    assert cells["brightness"][0] == pytest.approx(1.0, abs=1e-4)
    assert cells["brightness"][1] == pytest.approx(4.0, abs=1e-4)
    assert np.isnan(cells["brightness"][2])


def test_a_missing_optional_dependency_names_its_extra():
    with pytest.raises(
        media.MissingOptionalDependencyError, match="uv sync --extra labels-audio"
    ):
        media.require_module(
            "conv_wm_absent_extractor_backend",
            label="prosody.ego_f0_hz",
            extra="labels-audio",
        )


def test_the_base_package_imports_without_heavy_extras():
    import subprocess
    import sys

    code = (
        "import sys; import conv_wm.cli, conv_wm.data.labels.store, "
        "conv_wm.data.pipeline.labels, conv_wm.data.labels.media; "
        "heavy = {'parselmouth', 'mediapipe', 'mmpose', 'whisperx', 'pyannote', 'opensmile', "
        "'torch'}; print(sorted(heavy & set(sys.modules)))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert output.stdout.strip() == "[]"


def _fake_files(path: Path):
    def resolve(cfg, dataset, grid_sha256, stream):
        del cfg, dataset, grid_sha256, stream
        return {"r1": media.MediaFile(path, 0.0)}, {
            "media_manifest": {"file": "m", "sha256": "s"}
        }

    return resolve


def test_audio_extractor_with_praat_f0(monkeypatch, tmp_path):
    pytest.importorskip("parselmouth")
    structure = overlap_structure()
    audio = sine(150.0, 3.0, 0.3)
    path = tmp_path / "r1.wav"
    path.write_bytes(b"x")
    monkeypatch.setattr(media, "media_files", _fake_files(path))
    keys = GridKeys(np.arange(30), np.arange(30) * STEP)
    result = media.media_tables(
        None,  # type: ignore[arg-type]
        "toy",
        Extractor.AUDIO,
        {"r1": structure},
        {"r1": keys},
        LabelConfig(),
        grid_sha256="g",
        external=True,
        decode=lambda p, start, duration, rate: audio[
            int(start * rate) : int((start + duration) * rate)
        ],
    )
    grid = result.tables[Table.GRID].to_pandas()
    assert "prosody.ego_f0_hz" in result.materialized
    assert result.tools["praat"]["package_version"]
    assert grid.ego_f0_hz[11:14].tolist() == pytest.approx([150.0] * 3, rel=0.02)
    assert grid.ego_f0_hz[16:19].isna().all()  # overlap is never attributed
    assert grid.ego_f0_hz[3:8].isna().all()  # the wearer is silent
    assert grid.ego_voiced_fraction[12] == pytest.approx(1.0)
    assert grid.audio_valid.all()


def test_audio_extractor_without_external_builds_no_external_label(
    monkeypatch, tmp_path
):
    structure = overlap_structure()
    path = tmp_path / "r1.wav"
    path.write_bytes(b"x")
    monkeypatch.setattr(media, "media_files", _fake_files(path))
    keys = GridKeys(np.arange(30), np.arange(30) * STEP)
    result = media.media_tables(
        None,  # type: ignore[arg-type]
        "toy",
        Extractor.AUDIO,
        {"r1": structure},
        {"r1": keys},
        LabelConfig(),
        grid_sha256="g",
        external=False,
        decode=lambda p, start, duration, rate: sine(150.0, 3.0),
    )
    assert "prosody.ego_f0_hz" not in result.materialized
    assert "ego_f0_hz" not in result.tables[Table.GRID].column_names
    assert "praat" not in result.tools


def test_a_recording_without_media_is_invalid_not_silent(monkeypatch):
    structure = overlap_structure()

    def resolve(cfg, dataset, grid_sha256, stream):
        return {"r1": media.MediaFile(None, 0.0)}, {}

    monkeypatch.setattr(media, "media_files", resolve)
    keys = GridKeys(np.arange(30), np.arange(30) * STEP)
    result = media.media_tables(
        None,  # type: ignore[arg-type]
        "toy",
        Extractor.AUDIO,
        {"r1": structure},
        {"r1": keys},
        LabelConfig(),
        grid_sha256="g",
        external=False,
        decode=lambda *a: pytest.fail("nothing to decode"),
    )
    grid = result.tables[Table.GRID].to_pandas()
    assert not grid.audio_valid.any()
    assert grid.global_audio_rms.isna().all()
    assert result.statistics["recordings_without_media_file"] == 1
