from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFprobeError(RuntimeError):
    pass


def get_ffprobe_path() -> str:
    """Return the ffprobe executable available on PATH."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise FFprobeError("ffprobe executable not found in PATH")
    return ffprobe_path


def probe_json(path: Path, arguments: list[str]) -> dict:
    """Run ffprobe arguments for one path and return its JSON payload."""
    command = [
        get_ffprobe_path(),
        "-v",
        "error",
        *arguments,
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise FFprobeError(f"ffprobe failed for {path}: {result.stderr.strip()}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFprobeError(f"ffprobe returned invalid JSON for {path}") from exc


def probe_media(path: Path) -> dict:
    return probe_json(path, ["-show_format", "-show_streams"])
