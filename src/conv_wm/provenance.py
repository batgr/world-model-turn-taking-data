"""Execution provenance recorded in every generated report summary.

FFmpeg behaviour (AAC priming, ``skip_samples``, edit lists, ``-read_intervals``)
changes between releases, and reports produced from uncommitted code must be
distinguishable from committed ones. Every summary therefore records the
versions and repository state that produced it.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

PACKAGE_NAME = "conv_wm"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReportProvenance:
    """Who, when and with which tools a report was generated."""

    generated_at_utc: str
    """ISO-8601 UTC timestamp of report generation."""
    conv_wm_version: str
    """Installed package version from package metadata."""
    git_commit: str | None
    """Full HEAD commit hash, or ``None`` outside a git checkout."""
    git_dirty: bool | None
    """``True`` when tracked files had uncommitted changes."""
    python_version: str
    ffmpeg_version: str | None
    """First line of ``ffmpeg -version``; ``None`` when the executable is missing."""
    ffprobe_version: str | None
    """First line of ``ffprobe -version``; ``None`` when the executable is missing."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


def package_version(package: str = PACKAGE_NAME) -> str:
    """Return the installed version of ``package`` (``"unknown"`` if absent)."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    """Run ``command`` and return its stripped stdout, or ``None`` on any failure."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state(repo_root: Path = PROJECT_ROOT) -> tuple[str | None, bool | None]:
    """Return ``(commit_hash, dirty)`` for ``repo_root``; ``(None, None)`` if not git."""
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if not commit:
        return None, None
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root
    )
    if status is None:
        return commit, None
    return commit, bool(status)


def executable_version(name: str) -> str | None:
    """Return the first line of ``<name> -version``, or ``None`` when unavailable."""
    output = _run([name, "-version"])
    if not output:
        return None
    return output.splitlines()[0].strip()


def collect_provenance(repo_root: Path = PROJECT_ROOT) -> ReportProvenance:
    """Collect provenance from the live environment.

    Missing external executables are reported as ``None`` rather than raising, so
    audits that do not need FFmpeg still produce a summary.
    """
    commit, dirty = git_state(repo_root)
    return ReportProvenance(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        conv_wm_version=package_version(),
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        ffmpeg_version=executable_version("ffmpeg"),
        ffprobe_version=executable_version("ffprobe"),
    )
