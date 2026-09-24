"""End-to-end label sidecars on a synthetic EgoCom-shaped corpus, through the CLI.

Starts from cleaned annotation tables — the real native focal voice-state
build, control, grid, media manifest, model-ready and label stages all run —
so the label stage is checked against artifacts it did not hand-craft.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "data"))

from state_layers import config_for, long_timeline, write_egocom_interim

from conv_wm import cli
from conv_wm.data.labels import store
from conv_wm.data.labels.selection import LabelSelection, LabelUnavailableError
from conv_wm.data.pipeline.labels import dataset_dir, label_status

DATASET = "egocom"
RECORDINGS = {"rec-a": "train", "rec-b": "val", "rec-c": "test"}


def _corpus(tmp_path: Path, **overrides) -> tuple[DictConfig, Path]:
    cfg = config_for(tmp_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides)) if overrides else cfg
    assert isinstance(cfg, DictConfig)
    write_egocom_interim(
        cfg,
        {
            recording: (f"part-{recording}", split, long_timeline())
            for recording, split in RECORDINGS.items()
        },
        others={
            f"part-{recording}": [("1", 2.0, 2.4), ("1", 3.8, 5.0), ("3", 10.0, 10.5)]
            for recording in RECORDINGS
        },
    )
    manifest = Path(cfg.manifest.output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"relative_path": ["x"]}).to_parquet(manifest, index=False)
    path = tmp_path / "config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg))
    return cfg, path


def _run(config_path: Path, *args: str) -> int:
    return cli.main(["--config", str(config_path), *args])


@pytest.fixture
def built(tmp_path):
    cfg, config_path = _corpus(tmp_path)
    assert _run(config_path, "build", "all", "--dataset", DATASET) == cli.EXIT_OK
    return cfg, config_path


def test_build_all_writes_the_default_extractors(built):
    cfg, _ = built
    root = dataset_dir(cfg, DATASET)
    assert (root / "registry.json").exists()
    speech = store.read_manifest(root, "speech")
    assert speech is not None and speech["materialized_labels"]
    assert {"grid", "events", "segments", "participants", "recordings"} <= set(
        speech["tables"]
    )
    social = store.read_manifest(root, "social")
    assert social is not None and social["materialized_labels"] == []
    assert "social_native.looking_at_wearer_subframes" in social["unavailable_labels"]
    text = store.read_manifest(root, "text")
    assert text is not None and "text.speech_rate" in text["materialized_labels"]
    assert store.read_manifest(root, "audio") is None  # media extractors are opt-in


def test_the_label_grid_is_the_action_grid(built):
    cfg, _ = built
    grid = pd.read_parquet(
        Path(cfg.paths.processed)
        / "vocal_action_grid"
        / DATASET
        / "vocal_action_grid.parquet",
        columns=["recording_id", "decision_index", "decision_time_s"],
    )
    labels = pq.read_table(
        dataset_dir(cfg, DATASET) / "speech" / "grid.parquet",
        columns=["recording_id", "decision_index", "decision_time_s"],
    ).to_pandas()
    pd.testing.assert_frame_equal(
        grid.sort_values(["recording_id", "decision_index"]).reset_index(drop=True),
        labels,
        check_dtype=False,
    )


def test_every_participant_is_kept_with_its_own_id(built):
    cfg, _ = built
    recordings = pq.read_table(
        dataset_dir(cfg, DATASET) / "speech" / "recordings.parquet"
    )
    assert recordings.column("participant_ids").to_pylist()[0] == ["0", "1", "3"]


def test_the_wearer_agrees_with_the_native_focal_voice_state(built):
    cfg, _ = built
    report = json.loads(
        (
            Path(cfg.paths.reports) / "labels" / DATASET / "speech" / "report.json"
        ).read_text()
    )
    agreement = report["native_focal_voice_state_agreement"]
    assert agreement["recordings_compared"] == 3
    assert agreement["recordings_mismatched"] == 0
    assert report["lineage_chain"][-1] == "label sidecars"


def test_selections_load_only_what_they_ask_for(built):
    cfg, _ = built
    root = dataset_dir(cfg, DATASET)
    assert not store.load_labels(root, LabelSelection())
    exact = store.load_labels(
        root,
        LabelSelection(
            True,
            (
                "instantaneous.speaker_activity",
                "timing.time_to_next_ego_onset",
                "turns.floor_transfers",
            ),
        ),
    )
    assert set(exact.tables) == {"grid", "events"}
    assert set(exact["grid"].column_names) == {
        "recording_id",
        "decision_index",
        "decision_time_s",
        "participant_ids",
        "speaker_activity",
        "time_to_next_ego_onset",
        "time_to_next_ego_onset_valid",
    }
    assert set(exact["events"].column("event_type").to_pylist()) == {"floor_change"}
    audio = store.load_labels(root, LabelSelection(True, ("all",), ("audio",)))
    assert not any(name.startswith("text.") for name in audio.labels)
    assert "nuisance.global_audio_rms" in audio.skipped  # audio extractor not built
    families = store.load_labels(root, LabelSelection(True, ("timing.*", "events.*")))
    assert all(name.split(".")[0] in {"timing", "events"} for name in families.labels)
    with pytest.raises(LabelUnavailableError):
        store.load_labels(
            root, LabelSelection(True, ("social_native.face_tracked_subframes",))
        )


def test_a_rebuild_is_byte_identical(built):
    cfg, config_path = built
    root = dataset_dir(cfg, DATASET)
    before = {
        path.relative_to(root): store.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert _run(config_path, "build", "labels", "--dataset", DATASET) == cli.EXIT_OK
    after = {
        path.relative_to(root): store.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_a_config_change_marks_the_labels_stale(built, tmp_path, capsys):
    cfg, _ = built
    assert all(entry["current"] for entry in label_status(cfg, DATASET).values())
    changed = OmegaConf.merge(cfg, {"labels": {"future_horizons_s": [0.2, 2.0]}})
    assert isinstance(changed, DictConfig)
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(OmegaConf.to_yaml(changed))
    code = _run(changed_path, "labels", "status", "--dataset", DATASET)
    assert code == cli.EXIT_AUDIT_FAILED
    assert "label configuration changed" in capsys.readouterr().out
    assert _run(changed_path, "build", "labels", "--dataset", DATASET) == cli.EXIT_OK
    manifest = store.read_manifest(dataset_dir(changed, DATASET), "speech")
    assert manifest is not None and manifest["config"]["future_horizons_s"] == [
        0.2,
        2.0,
    ]


def test_a_changed_upstream_annotation_is_rejected(built, capsys):
    cfg, config_path = built
    transcript = Path(cfg.datasets.egocom.interim) / "ground_truth_clean.parquet"
    table = pd.read_parquet(transcript)
    table["endTime"] = table["endTime"].astype(float) + 0.5
    table.to_parquet(transcript, index=False)
    code = _run(config_path, "build", "labels", "--dataset", DATASET)
    assert code == cli.EXIT_FAILURE
    assert "changed after the action grid was built" in capsys.readouterr().err


def test_a_tampered_table_is_reported_stale(built):
    cfg, _ = built
    path = dataset_dir(cfg, DATASET) / "speech" / "grid.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 10), path)
    status = label_status(cfg, DATASET)["speech"]
    assert not status["current"] and "checksum" in status["reasons"][0]


def test_the_coverage_audit_reports_every_label(built):
    cfg, config_path = built
    assert _run(config_path, "audit", "labels", "--dataset", DATASET) == cli.EXIT_OK
    directory = Path(cfg.paths.reports) / "labels" / "coverage"
    report = json.loads((directory / f"coverage_{DATASET}.json").read_text())
    entry = report["datasets"][DATASET]["labels"]
    assert entry["instantaneous.ego_speaking"]["materialized"]
    assert entry["instantaneous.ego_speaking"]["valid"] == pytest.approx(1.0)
    assert entry["social_states.dominance"]["supported"] is False
    sanity = report["datasets"][DATASET]["sanity"]
    assert sanity["ego_onset_count"] > 0 and sanity["floor_transfer_count"] > 0
    assert (
        (directory / f"coverage_{DATASET}.md")
        .read_text()
        .startswith("# Label coverage")
    )


def test_labels_are_built_only_on_request_for_media_extractors(built, capsys):
    _, config_path = built
    code = _run(
        config_path, "build", "labels", "--dataset", DATASET, "--labels", "text.*"
    )
    assert code == cli.EXIT_OK
    assert "egocom/text" in capsys.readouterr().out
