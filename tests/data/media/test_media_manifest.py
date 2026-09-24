"""The media manifest contract: one portable media record per canonical recording."""

from __future__ import annotations

import pandas as pd
import pytest

from conv_wm.data.media.media_manifest import (
    MEDIA_MANIFEST_COLUMNS,
    AmbiguousMediaError,
    MediaFile,
    MediaFileIndex,
    MediaManifestError,
    MediaRecord,
    MediaSource,
    is_portable_path,
    media_coverage,
    media_manifest_table,
    validate_media_manifest,
)

VIDEO = MediaFile("Corpus/videos/rec-1.mp4", has_audio_stream=True)
AUDIO = MediaFile("Corpus/audio/rec-1.wav", has_audio_stream=None)
CANONICAL = {("corpus", "rec-1"), ("corpus", "rec-2"), ("corpus", "rec-3")}


def _table(*records: MediaRecord) -> pd.DataFrame:
    table, _ = media_manifest_table(MediaSource("corpus", records))
    return table


def _row(**overrides) -> pd.DataFrame:
    row = {
        "dataset": "corpus",
        "recording_id": "rec-1",
        "video_path": "videos/rec-1.mp4",
        "audio_path": None,
        "media_offset_s": 0.0,
        "video_has_audio": True,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=list(MEDIA_MANIFEST_COLUMNS))


def test_a_valid_record_passes_and_keeps_the_canonical_schema():
    table = _table(MediaRecord("rec-1", VIDEO, AUDIO, media_offset_s=12.5))

    validate_media_manifest(table, CANONICAL)
    assert tuple(table.columns) == MEDIA_MANIFEST_COLUMNS
    assert table.iloc[0]["media_offset_s"] == 12.5


def test_video_only_record_is_valid():
    table = _table(MediaRecord("rec-1", VIDEO, None))

    validate_media_manifest(table, CANONICAL)
    assert table.iloc[0]["video_path"] == "videos/rec-1.mp4"
    assert pd.isna(table.iloc[0]["audio_path"])
    assert bool(table.iloc[0]["video_has_audio"]) is True


def test_audio_only_record_is_valid():
    table = _table(MediaRecord("rec-1", None, AUDIO))

    validate_media_manifest(table, CANONICAL)
    assert pd.isna(table.iloc[0]["video_path"])
    assert table.iloc[0]["audio_path"] == "audio/rec-1.wav"
    assert pd.isna(table.iloc[0]["video_has_audio"])


def test_audio_and_video_is_one_row_not_two():
    table = _table(MediaRecord("rec-1", VIDEO, AUDIO))

    validate_media_manifest(table, CANONICAL)
    assert len(table) == 1
    assert table.iloc[0][["video_path", "audio_path"]].notna().all()


def test_a_record_with_neither_modality_is_rejected():
    with pytest.raises(MediaManifestError, match="neither video_path nor audio_path"):
        validate_media_manifest(_row(video_path=None, audio_path=None), CANONICAL)


@pytest.mark.parametrize(
    "path",
    [
        "/Volumes/PortableSSD/data/raw/EgoCom/x.mp4",
        "/Users/someone/x.mp4",
        "~/x.mp4",
        "C:/data/x.mp4",
        "videos\\x.mp4",
        "../outside/x.mp4",
        "",
    ],
)
def test_absolute_or_escaping_paths_are_rejected(path):
    assert not is_portable_path(path)
    with pytest.raises(MediaManifestError, match="non-portable video_path"):
        validate_media_manifest(_row(video_path=path), CANONICAL)


def test_duplicate_recording_mappings_are_rejected():
    table = pd.concat([_row(), _row(video_path="other/rec-1.mp4")], ignore_index=True)

    with pytest.raises(MediaManifestError, match="ambiguous"):
        validate_media_manifest(table, CANONICAL)


def test_an_unknown_canonical_recording_is_rejected():
    with pytest.raises(MediaManifestError, match="unknown to the canonical data"):
        validate_media_manifest(_row(recording_id="rec-404"), CANONICAL)


def test_a_null_identity_is_rejected():
    with pytest.raises(MediaManifestError, match="null recording_id"):
        validate_media_manifest(_row(recording_id=None), CANONICAL)


def test_relative_paths_are_kept_below_the_corpus_directory():
    nested = MediaFile("Corpus/240p/20min/Part_1/rec-1.MP4", has_audio_stream=True)

    table, directory = media_manifest_table(
        MediaSource("corpus", (MediaRecord("rec-1", nested, None),))
    )

    assert directory == "Corpus"
    assert table.iloc[0]["video_path"] == "240p/20min/Part_1/rec-1.MP4"
    assert is_portable_path(table.iloc[0]["video_path"])


def test_a_path_outside_the_dataset_corpus_directory_is_rejected():
    stray = MediaFile("OtherCorpus/rec-1.mp4", has_audio_stream=True)

    with pytest.raises(MediaManifestError, match="not below"):
        media_manifest_table(
            MediaSource("corpus", (MediaRecord("rec-1", stray, None),))
        )


def test_the_file_index_pairs_video_and_audio_by_stem():
    media = pd.DataFrame(
        {
            "dataset": ["corpus", "corpus", "corpus", "other"],
            "relative_path": [
                "Corpus/v/rec-1.mp4",
                "Corpus/a/rec-1.wav",
                "Corpus/v/rec-1.json",
                "Other/v/rec-2.mp4",
            ],
            "probe_ok": [True, True, True, True],
            "n_audio_streams": [0, 1, None, 1],
        }
    )
    index = MediaFileIndex(media, "corpus")

    video, audio = index.resolve("rec-1")
    assert video == MediaFile("Corpus/v/rec-1.mp4", has_audio_stream=False)
    assert audio is not None and audio.relative_path == "Corpus/a/rec-1.wav"
    assert index.resolve("rec-2") == (None, None)


def test_the_file_index_refuses_to_pick_between_two_files():
    media = pd.DataFrame(
        {
            "dataset": ["corpus", "corpus"],
            "relative_path": ["Corpus/240p/rec-1.mp4", "Corpus/720p/rec-1.mp4"],
            "probe_ok": [True, True],
            "n_audio_streams": [1, 1],
        }
    )

    with pytest.raises(AmbiguousMediaError, match="several video files"):
        MediaFileIndex(media, "corpus").resolve("rec-1")


def test_coverage_counts_modalities_and_flags_labelled_unresolved_recordings():
    table = _table(
        MediaRecord("rec-1", VIDEO, AUDIO),
        MediaRecord("rec-2", MediaFile("Corpus/v/rec-2.mp4", False), None),
    )

    coverage = media_coverage(
        table,
        ["rec-1", "rec-2", "rec-3", "rec-4"],
        {"rec-3": "media_missing", "rec-4": "media_missing"},
        {"rec-1": 10, "rec-2": 10, "rec-3": 0, "rec-4": 5},
    )

    assert coverage["recordings_in_action_grid"] == 4
    assert coverage["recordings_in_media_manifest"] == 2
    assert coverage["video_available"] == 2
    assert coverage["audio_path_available"] == 1
    assert coverage["both"] == 1
    assert coverage["video_only"] == 1
    assert coverage["audio_only"] == 0
    assert coverage["video_with_embedded_audio"] == 1
    assert coverage["recordings_without_usable_audio"] == 1
    assert coverage["recordings_without_resolved_media"] == 2
    assert coverage["unresolved_reasons"] == {"media_missing": 2}
    assert coverage["unresolved_with_valid_slots"] == ["rec-4"]
