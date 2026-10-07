"""The label store: atomic layout, manifests and staleness."""

from __future__ import annotations

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
    manifest = store.read_manifest(dataset_dir, Extractor.SPEECH)
    assert manifest is not None
    with pytest.raises(store.StaleLabelsError, match="checksum"):
        store.verify_extractor(dataset_dir, manifest)


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
