from pathlib import Path

import pandas as pd
import pytest

from conv_wm.data.manifest import (
    MANIFEST_COLUMNS,
    _classify_file_type,
    _compute_sha256,
    _iter_raw_files,
    build_manifest,
    save_manifest,
    validate_manifest,
)


def test_classify_video_file():
    assert _classify_file_type(Path("clip.mp4")) == "video"


def test_classify_audio_file():
    assert _classify_file_type(Path("audio.wav")) == "audio"


def test_classify_tabular_file():
    assert _classify_file_type(Path("data.csv")) == "tabular"


def test_classify_structured_data_file():
    assert _classify_file_type(Path("annotations.json")) == "structured_data"


def test_classify_unknown_file():
    assert _classify_file_type(Path("README.xyz")) == "other"


def test_classification_is_case_insensitive():
    assert _classify_file_type(Path("CLIP.MP4")) == "video"


def test_iter_raw_files_returns_sorted_files(tmp_path: Path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")

    files = list(_iter_raw_files(tmp_path))

    assert [p.name for p in files] == ["a.txt", "b.txt"]


def test_iter_raw_files_is_recursive(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()

    (nested / "file.txt").write_text("content")

    files = list(_iter_raw_files(tmp_path))

    assert files == [nested / "file.txt"]


def test_iter_raw_files_ignores_system_files(tmp_path: Path):
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "valid.txt").write_text("ok")

    files = list(_iter_raw_files(tmp_path))

    assert files == [tmp_path / "valid.txt"]


def test_iter_raw_files_raises_if_root_missing(tmp_path: Path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        list(_iter_raw_files(missing))


def test_compute_sha256_is_deterministic(tmp_path: Path):
    file = tmp_path / "sample.txt"
    file.write_text("hello")

    hash_1 = _compute_sha256(file)
    hash_2 = _compute_sha256(file)

    assert hash_1 == hash_2
    assert len(hash_1) == 64


def test_compute_sha256_changes_with_content(tmp_path: Path):
    file = tmp_path / "sample.txt"

    file.write_text("hello")
    hash_1 = _compute_sha256(file)

    file.write_text("world")
    hash_2 = _compute_sha256(file)

    assert hash_1 != hash_2


def test_build_manifest(tmp_path: Path):
    raw_root = tmp_path / "raw"

    egocom = raw_root / "EgoCom"
    ego4d = raw_root / "Ego4D"

    egocom.mkdir(parents=True)
    ego4d.mkdir(parents=True)

    (egocom / "video.mp4").write_bytes(b"video")
    (egocom / "metadata.csv").write_text("a,b\n1,2\n")
    (ego4d / "annotations.json").write_text("{}")

    manifest = build_manifest(raw_root)

    assert len(manifest) == 3
    assert set(manifest["dataset"]) == {"egocom", "ego4d"}


def test_build_manifest_with_checksums(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    (raw_root / "file.csv").write_text("hello")

    manifest = build_manifest(
        raw_root,
        compute_checksum=True,
    )

    assert manifest.loc[0, "checksum"] is not None
    assert manifest.loc[0, "checksum_algorithm"] == "sha256"


def test_save_manifest(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    (raw_root / "sample.csv").write_text("a,b\n1,2\n")

    manifest = build_manifest(raw_root)

    output_path = tmp_path / "reports" / "raw_manifest.parquet"

    save_manifest(manifest, output_path)

    assert output_path.exists()

    loaded = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(
        loaded,
        manifest,
        check_dtype=True,
    )


def test_iter_raw_files_ignores_appledouble_files(tmp_path: Path):
    (tmp_path / "._video.mp4").write_bytes(b"metadata")
    (tmp_path / "video.mp4").write_bytes(b"video")

    files = list(_iter_raw_files(tmp_path))

    assert files == [tmp_path / "video.mp4"]


def _valid_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "egocom",
                "relative_path": "EgoCom/video.mp4",
                "file_name": "video.mp4",
                "extension": ".mp4",
                "file_type": "video",
                "size_bytes": 100,
                "checksum": None,
                "checksum_algorithm": None,
            }
        ],
        columns=MANIFEST_COLUMNS,
    )


def test_validate_manifest_accepts_valid_manifest():
    manifest = _valid_manifest()

    validate_manifest(manifest)


def test_validate_manifest_rejects_missing_columns():
    manifest = _valid_manifest().drop(columns=["size_bytes"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_duplicates():
    manifest = pd.concat(
        [_valid_manifest(), _valid_manifest()],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_negative_file_size():
    manifest = _valid_manifest()
    manifest.loc[0, "size_bytes"] = -1

    with pytest.raises(ValueError, match="negative file sizes"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_empty_relative_path():
    manifest = _valid_manifest()
    manifest.loc[0, "relative_path"] = ""

    with pytest.raises(ValueError, match="empty relative paths"):
        validate_manifest(manifest)
