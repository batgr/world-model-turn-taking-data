import json
import subprocess
from pathlib import Path

import pytest

from conv_wm import provenance
from conv_wm.reports import write_summary


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


@pytest.fixture
def fake_environment(monkeypatch):
    def fake_run(command, **kwargs):
        program = command[0]
        if program == "git" and command[1] == "rev-parse":
            return _completed("abc123\n")
        if program == "git" and command[1] == "status":
            return _completed(" M src/x.py\n")
        if program == "ffmpeg":
            return _completed(
                "ffmpeg version 7.1 Copyright (c) 2000-2024\nbuilt with clang\n"
            )
        if program == "ffprobe":
            raise FileNotFoundError(program)
        raise AssertionError(command)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    monkeypatch.setattr(provenance.metadata, "version", lambda name: "9.9.9")


def test_collect_provenance_reads_live_tools(fake_environment):
    result = provenance.collect_provenance()

    assert result.conv_wm_version == "9.9.9"
    assert result.git_commit == "abc123"
    assert result.git_dirty is True
    assert result.ffmpeg_version == "ffmpeg version 7.1 Copyright (c) 2000-2024"
    assert result.ffprobe_version is None
    assert result.generated_at_utc.endswith("+00:00")


def test_missing_git_and_package_are_reported_not_raised(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda command, **kwargs: _completed("", returncode=128),
    )

    def missing(name):
        raise provenance.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "version", missing)

    result = provenance.collect_provenance()

    assert result.git_commit is None
    assert result.git_dirty is None
    assert result.conv_wm_version == "unknown"
    assert result.ffmpeg_version is None


def test_write_summary_stamps_provenance_and_parameters(
    tmp_path: Path, fake_environment
):
    path = tmp_path / "nested" / "summary.json"

    document = write_summary(
        path,
        {"n_files": 3, "datasets": {"a": 1}},
        parameters={"drift_threshold_ms": 1.0},
    )

    written = json.loads(path.read_text())
    assert written == document
    assert written["n_files"] == 3
    assert written["parameters"] == {"drift_threshold_ms": 1.0}
    assert written["provenance"]["git_commit"] == "abc123"
    assert written["provenance"]["git_dirty"] is True
    assert written["summary_schema_version"] == 1


def test_write_summary_rejects_reserved_keys(tmp_path: Path, fake_environment):
    with pytest.raises(ValueError, match="reserved"):
        write_summary(tmp_path / "s.json", {"provenance": {}})
