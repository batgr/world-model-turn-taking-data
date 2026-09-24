"""The label store: atomic layout, column projection, optional loading, staleness."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conv_wm.data.labels import store
from conv_wm.data.labels.registry import (
    LABEL_SCHEMA_VERSION,
    REGISTRY_VERSION,
    Extractor,
    Table,
)
from conv_wm.data.labels.selection import LabelSelection, LabelUnavailableError


def _grid(values: list[float], *, recordings=("a", "b")) -> pa.Table:
    rows = len(values)
    return pa.table(
        {
            "recording_id": [
                recordings[i * len(recordings) // rows] for i in range(rows)
            ],
            "decision_index": list(range(rows)),
            "decision_time_s": [0.1 * i for i in range(rows)],
            "ego_speaking": [v > 0 for v in values],
            "time_to_next_ego_onset": values,
            "time_to_next_ego_onset_valid": [True] * rows,
            "participant_ids": [["w", "x"]] * rows,
            "speaker_activity": [[True, False]] * rows,
        }
    )


def _events() -> pa.Table:
    return pa.table(
        {
            "recording_id": ["a", "a", "b"],
            "event_type": ["onset", "offset", "onset"],
            "time_s": [0.05, 0.25, 0.45],
            "participant_index": [0, 0, 1],
            "participant_id": ["w", "w", "x"],
            "is_ego": [True, True, False],
        }
    )


def _manifest(extractor: str, materialized: list[str], grid_sha: str = "g1") -> dict:
    return {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "dataset": "toy",
        "extractor": extractor,
        "materialized_labels": materialized,
        "unavailable_labels": {"events.ego_onset_subframes": "not built in this toy"},
        "inputs": {"action_grid": {"sha256": grid_sha}},
    }


@pytest.fixture
def dataset_dir(tmp_path):
    root = tmp_path / "labels" / "toy"
    store.write_registry(root)
    store.write_extractor(
        root / "speech",
        {Table.GRID: _grid([0.0, 0.3, 0.2, 0.0]), Table.EVENTS: _events()},
        _manifest(
            "speech",
            [
                "instantaneous.ego_speaking",
                "instantaneous.speaker_activity",
                "timing.time_to_next_ego_onset",
                "events.speaker_offsets",
            ],
        ),
    )
    return root


def test_disabled_labels_touch_no_file(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a label file was read")

    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(store, "read_manifest", forbidden)
    bundle = store.load_labels(tmp_path / "does-not-exist", LabelSelection())
    assert not bundle and bundle.labels == ()


def test_only_the_requested_columns_are_read(dataset_dir, monkeypatch):
    calls = []
    original = pq.read_table

    def spy(path, *args, **kwargs):
        calls.append((path.name, tuple(kwargs["columns"])))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    bundle = store.load_labels(
        dataset_dir, LabelSelection(True, ("timing.time_to_next_ego_onset",))
    )
    assert calls == [
        (
            "grid.parquet",
            (
                "recording_id",
                "decision_index",
                "decision_time_s",
                "time_to_next_ego_onset",
                "time_to_next_ego_onset_valid",
            ),
        )
    ]
    assert set(bundle.tables) == {"grid"}
    assert bundle["grid"].column_names == list(calls[0][1])


def test_participant_axis_labels_bring_the_participant_ids(dataset_dir):
    bundle = store.load_labels(
        dataset_dir, LabelSelection(True, ("instantaneous.speaker_activity",))
    )
    assert "participant_ids" in bundle["grid"].column_names


def test_events_are_filtered_to_the_requested_event_types(dataset_dir):
    bundle = store.load_labels(
        dataset_dir, LabelSelection(True, ("events.speaker_offsets",))
    )
    events = bundle["events"]
    assert set(events.column("event_type").to_pylist()) == {"offset"}


def test_recording_and_window_filters(dataset_dir):
    bundle = store.load_labels(
        dataset_dir,
        LabelSelection(True, ("instantaneous.ego_speaking", "events.speaker_offsets")),
        recording_ids=["a"],
        start_s=0.1,
        end_s=0.3,
    )
    grid = bundle["grid"]
    assert grid.column("recording_id").to_pylist() == ["a"]
    assert grid.column("decision_index").to_pylist() == [1]
    assert bundle["events"].column("time_s").to_pylist() == [0.25]


def test_a_label_the_store_did_not_materialize_is_unavailable_when_named(dataset_dir):
    with pytest.raises(LabelUnavailableError, match="not built in this toy"):
        store.load_labels(
            dataset_dir, LabelSelection(True, ("events.ego_onset_subframes",))
        )


def test_an_extractor_never_built_is_skipped_under_a_wildcard(dataset_dir):
    bundle = store.load_labels(dataset_dir, LabelSelection(True, ("nuisance.*",)))
    assert not bundle.tables
    assert all("was not built" in r for r in bundle.skipped.values())


def test_an_extractor_never_built_is_an_error_when_named(dataset_dir):
    with pytest.raises(LabelUnavailableError, match="--extractors audio"):
        store.load_labels(
            dataset_dir, LabelSelection(True, ("nuisance.global_audio_rms",))
        )


def test_all_loads_what_exists_and_reports_the_rest(dataset_dir):
    bundle = store.load_labels(dataset_dir, LabelSelection.everything())
    assert set(bundle.labels) == {
        "instantaneous.ego_speaking",
        "instantaneous.speaker_activity",
        "timing.time_to_next_ego_onset",
        "events.speaker_offsets",
    }
    assert "social_states.dominance" in bundle.skipped


def test_the_manifest_records_checksums_rows_and_columns(dataset_dir):
    manifest = store.read_manifest(dataset_dir, Extractor.SPEECH)
    assert manifest is not None
    grid = manifest["tables"]["grid"]
    assert grid["rows"] == 4
    assert grid["sha256"] == store.sha256_file(dataset_dir / "speech" / "grid.parquet")
    assert "ego_speaking" in grid["columns"]


def test_a_tampered_table_is_rejected_when_verifying(dataset_dir):
    path = dataset_dir / "speech" / "grid.parquet"
    pq.write_table(_grid([1.0, 1.0, 1.0, 1.0]), path)
    with pytest.raises(store.StaleLabelsError, match="checksum"):
        store.load_labels(
            dataset_dir,
            LabelSelection(True, ("instantaneous.ego_speaking",)),
            verify=True,
        )


def test_a_manifest_from_another_registry_version_is_rejected(dataset_dir):
    path = dataset_dir / "speech" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["registry_version"] = REGISTRY_VERSION + 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(store.StaleLabelsError, match="registry"):
        store.load_labels(
            dataset_dir, LabelSelection(True, ("instantaneous.ego_speaking",))
        )


def test_extractors_built_from_different_grids_are_rejected(dataset_dir):
    store.write_extractor(
        dataset_dir / "audio",
        {
            Table.GRID: _grid([0.0, 0.0, 0.0, 0.0])
            .select(["recording_id", "decision_index", "decision_time_s"])
            .append_column("global_audio_rms", pa.array([0.1] * 4))
        },
        _manifest("audio", ["nuisance.global_audio_rms"], grid_sha="g2"),
    )
    with pytest.raises(store.StaleLabelsError, match="different action grids"):
        store.load_labels(
            dataset_dir,
            LabelSelection(
                True, ("nuisance.global_audio_rms", "instantaneous.ego_speaking")
            ),
        )


def test_grids_of_two_extractors_are_joined_on_their_keys(dataset_dir):
    audio_grid = (
        _grid([0.0] * 4)
        .select(["recording_id", "decision_index", "decision_time_s"])
        .append_column("global_audio_rms", pa.array([0.1, 0.2, 0.3, 0.4], pa.float32()))
    )
    store.write_extractor(
        dataset_dir / "audio",
        {Table.GRID: audio_grid},
        _manifest("audio", ["nuisance.global_audio_rms"]),
    )
    bundle = store.load_labels(
        dataset_dir,
        LabelSelection(
            True, ("nuisance.global_audio_rms", "instantaneous.ego_speaking")
        ),
    )
    grid = bundle["grid"]
    assert {"ego_speaking", "global_audio_rms"} <= set(grid.column_names)
    assert grid.num_rows == 4


def test_rewriting_an_extractor_replaces_it_whole(dataset_dir):
    store.write_extractor(
        dataset_dir / "speech",
        {Table.GRID: _grid([0.0, 0.0])},
        _manifest("speech", ["instantaneous.ego_speaking"]),
    )
    assert not (dataset_dir / "speech" / "events.parquet").exists()
    assert not list(dataset_dir.glob(".speech*"))


def test_a_failed_write_leaves_the_previous_extractor_intact(dataset_dir):
    before = (dataset_dir / "speech" / "manifest.json").read_text()

    with pytest.raises(AttributeError):
        store.write_extractor(
            dataset_dir / "speech",
            {Table.GRID: _grid([0.0]), Table.EVENTS: "not a table"},  # type: ignore[dict-item]
            _manifest("speech", []),
        )
    assert (dataset_dir / "speech" / "manifest.json").read_text() == before
    assert not list(dataset_dir.glob(".speech*"))
