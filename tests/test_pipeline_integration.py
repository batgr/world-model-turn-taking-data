"""End-to-end smoke test of the canonical pipeline, on a synthetic fixture.

Runs `conv-wm build all` through the real CLI on a tiny hand-written native
focal voice state, and checks that the model-ready index comes out the other
end. No corpus, no media, no ffprobe: this is the test a contributor runs to
know the wiring is intact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "data"))

from state_layers import (
    config_for,
    long_timeline,
    write_media_metadata,
    write_native_state,
    write_recording_splits,
)

from conv_wm import cli

DATASET = "egocom"
RECORDINGS = {"rec-a": "train", "rec-b": "val", "rec-c": "test"}


def _fixture(tmp_path: Path) -> tuple[DictConfig, Path]:
    """A three-recording synthetic corpus, already at the native-state stage.

    The first stage reads media metadata and cleaned annotations, which a
    synthetic fixture has none of, so the fixture starts one stage later.
    """
    cfg = config_for(tmp_path)
    frames = []
    for recording in RECORDINGS:
        # each call rewrites the single-recording timeline; keep a copy of each.
        # One conversation per recording, so the three splits are legitimate.
        write_native_state(
            cfg,
            DATASET,
            long_timeline(),
            recording=recording,
            conversation=f"conversation-{recording}",
        )
        frames.append(pd.read_parquet(_native_path(cfg)))
    _rewrite_native_state(cfg, pd.concat(frames, ignore_index=True))
    write_recording_splits(cfg, DATASET, RECORDINGS)
    write_media_metadata(
        cfg, [(DATASET, f"EgoCom/240p/{recording}.MP4", 1) for recording in RECORDINGS]
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))
    return cfg, config_path


def _native_path(cfg: DictConfig) -> Path:
    return (
        Path(cfg.paths.processed)
        / "native_focal_voice_state"
        / DATASET
        / "focal_voice_intervals.parquet"
    )


def _rewrite_native_state(cfg: DictConfig, timeline: pd.DataFrame) -> None:
    """Replace the fixture timeline and the checksum its report records."""
    path = _native_path(cfg)
    timeline.to_parquet(path, index=False)
    report_path = (
        Path(cfg.paths.reports) / "native_focal_voice_state" / DATASET / "report.json"
    )
    report = json.loads(report_path.read_text())
    report["output_artifacts"]["timeline"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    report_path.write_text(json.dumps(report))


def test_build_all_produces_a_model_ready_index(tmp_path, capsys):
    cfg, config_path = _fixture(tmp_path)

    code = cli.main(
        [
            "--config",
            str(config_path),
            "build",
            "all",
            "--dataset",
            DATASET,
            "--from",
            "control-focal-voice-state",
        ]
    )

    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Recordings: 3" in out

    index = pd.read_parquet(Path(cfg.paths.model_ready) / DATASET / "index.parquet")
    metadata = json.loads(
        (Path(cfg.paths.model_ready) / DATASET / "metadata.json").read_text()
    )

    assert set(index["recording_id"]) == set(RECORDINGS)
    assert index["future_steps"].eq(10).all()
    assert index["max_context_steps"].between(10, 50).all()
    assert set(index["sample_class"]) == {"event", "background"}
    # the upstream "val" spelling reaches the artifact as "validation"
    assert set(index["split"]) == {"train", "validation", "test"}
    assert index.groupby("recording_id")["split"].nunique().eq(1).all()
    assert index.groupby("conversation_id")["split"].nunique().eq(1).all()
    assert all(metadata["contract_checks"].values())
    assert metadata["splits"]["anchors"].keys() == {"train", "validation", "test"}


def test_build_all_ships_a_media_manifest_next_to_the_index(tmp_path):
    cfg, config_path = _fixture(tmp_path)

    code = cli.main(
        [
            "--config",
            str(config_path),
            "build",
            "all",
            "--dataset",
            DATASET,
            "--from",
            "control-focal-voice-state",
        ]
    )

    assert code == cli.EXIT_OK
    release = Path(cfg.paths.model_ready) / DATASET
    grid = pd.read_parquet(
        Path(cfg.paths.processed)
        / "vocal_action_grid"
        / DATASET
        / "vocal_action_grid.parquet"
    )
    media = pd.read_parquet(release / "media_manifest.parquet")
    metadata = json.loads((release / "metadata.json").read_text())
    report = json.loads(
        (
            Path(cfg.paths.reports) / "media_manifest" / DATASET / "report.json"
        ).read_text()
    )

    assert set(media["recording_id"]) == set(grid["recording_id"])
    assert media["video_path"].tolist() == [f"240p/{r}.MP4" for r in sorted(RECORDINGS)]
    assert media["audio_path"].isna().all()
    assert report["status"] == "PASS"
    assert report["lineage_chain"][-1] == "media manifest"
    assert report["media_schema_version"] == 1
    assert metadata["files"]["media_manifest"] == "media_manifest.parquet"
    assert metadata["media"]["corpus_directory"] == "EgoCom"
    assert metadata["media"]["raw_media_distributed"] is False
    assert str(tmp_path) not in json.dumps(metadata["media"])


def test_every_stage_writes_its_artifact_and_report(tmp_path):
    cfg, config_path = _fixture(tmp_path)

    cli.main(
        [
            "--config",
            str(config_path),
            "build",
            "all",
            "--dataset",
            DATASET,
            "--from",
            "control-focal-voice-state",
        ]
    )

    processed = Path(cfg.paths.processed)
    reports = Path(cfg.paths.reports)
    for relative in (
        f"control_focal_voice_state/{DATASET}/control_focal_voice_intervals.parquet",
        f"vocal_action_grid/{DATASET}/vocal_action_grid.parquet",
    ):
        assert (processed / relative).exists(), relative
    assert (Path(cfg.paths.model_ready) / DATASET / "index.parquet").exists()
    for stage in (
        "control_focal_voice_state",
        "vocal_action_grid",
        "media_manifest",
        "model_ready",
    ):
        report = json.loads((reports / stage / DATASET / "report.json").read_text())
        assert report["dataset"] == DATASET
        assert report["lineage_chain"][-1]


def test_one_conversation_split_across_splits_is_reported_as_leakage(tmp_path):
    """The same fixture with a single shared conversation must fail the check."""
    cfg = config_for(tmp_path)
    frames = []
    for recording in RECORDINGS:
        write_native_state(
            cfg, DATASET, long_timeline(), recording=recording, conversation="shared"
        )
        frames.append(pd.read_parquet(_native_path(cfg)))
    _rewrite_native_state(cfg, pd.concat(frames, ignore_index=True))
    write_recording_splits(cfg, DATASET, RECORDINGS)
    write_media_metadata(
        cfg, [(DATASET, f"EgoCom/240p/{recording}.MP4", 1) for recording in RECORDINGS]
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))

    cli.main(
        [
            "--config",
            str(config_path),
            "build",
            "all",
            "--dataset",
            DATASET,
            "--from",
            "control-focal-voice-state",
        ]
    )

    metadata = json.loads(
        (Path(cfg.paths.model_ready) / DATASET / "metadata.json").read_text()
    )
    report = json.loads(
        (Path(cfg.paths.reports) / "model_ready" / DATASET / "report.json").read_text()
    )
    assert metadata["contract_checks"]["no_recording_in_multiple_splits"] is False
    assert report["statistics"]["split_leakage"] == {
        "shared": ["test", "train", "validation"]
    }


def test_a_stage_run_out_of_order_names_the_command_that_fixes_it(tmp_path, capsys):
    cfg = config_for(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))

    code = cli.main(
        ["--config", str(config_path), "build", "model-ready", "--dataset", DATASET]
    )

    assert code == cli.EXIT_FAILURE
    assert "conv-wm build vocal-action-grid" in capsys.readouterr().err
