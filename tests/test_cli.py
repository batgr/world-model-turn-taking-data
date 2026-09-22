from pathlib import Path

import pytest
from omegaconf import OmegaConf

from conv_wm import cli
from conv_wm.data.media import ffprobe


def _config(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    cfg = {
        "paths": {
            stage: str(tmp_path / stage)
            for stage in ("raw", "interim", "validated", "processed", "model_ready")
        }
        | {"reports": str(reports)},
        "manifest": {
            "compute_checksum": False,
            "output": str(reports / "manifest" / "raw_manifest.parquet"),
        },
        "datasets": {},
    }
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def test_help_lists_every_audit(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["audit", "--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for name in (
        "manifest",
        "structure",
        "media",
        "video",
        "audio",
        "sync",
        "annotations",
        "vocal-annotation-coverage",
    ):
        assert name in out


def test_clean_help_lists_annotation_cleaning(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["clean", "--help"])

    assert exit_info.value.code == 0
    assert "annotations" in capsys.readouterr().out


def test_missing_prerequisite_exits_1_and_names_the_producing_command(tmp_path, capsys):
    code = cli.main(["--config", str(_config(tmp_path)), "audit", "sync"])

    assert code == cli.EXIT_FAILURE
    assert "conv-wm audit media" in capsys.readouterr().err


def test_missing_ffprobe_exits_1_with_a_clear_message(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ffprobe.shutil, "which", lambda name: None)
    config = _config(tmp_path)
    (tmp_path / "raw").mkdir()
    cli.main(["--config", str(config), "audit", "manifest"])

    code = cli.main(["--config", str(config), "audit", "media"])

    assert code == cli.EXIT_FAILURE
    assert "ffprobe" in capsys.readouterr().err


def test_manifest_audit_runs_end_to_end_on_an_empty_root(tmp_path, capsys):
    config = _config(tmp_path)
    (tmp_path / "raw" / "Synthetic").mkdir(parents=True)
    (tmp_path / "raw" / "Synthetic" / "a.wav").write_bytes(b"x")

    code = cli.main(["--config", str(config), "audit", "manifest"])

    assert code == cli.EXIT_OK
    assert (tmp_path / "reports" / "manifest" / "summary.json").exists()
    assert "1 files" in capsys.readouterr().out
