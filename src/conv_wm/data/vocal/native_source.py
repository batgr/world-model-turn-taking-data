"""What a dataset adapter hands to the native focal voice-state builder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.vocal.native_state import NativeAnnotation, NativeFocalRecording


@dataclass(frozen=True)
class NativeFocalVoiceSource:
    """Recordings of one corpus plus the provenance the report must carry."""

    dataset: str
    recordings: list[NativeFocalRecording]
    annotation_paths: tuple[Path, ...]
    """Interim annotation tables the recordings were derived from (checksummed)."""
    annotation_schema_version: str
    cleaning_rule_version: str
    statistics: dict[str, int] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class MediaCoverage:
    """Audio time range one probed media file provides, on its own timeline."""

    probe_ok: bool
    audio_start_s: float
    audio_end_s: float


def require_table(path: Path) -> pd.DataFrame:
    """Read an interim table or name the command that produces it."""
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with="conv-wm clean annotations")
    return pd.read_parquet(path)


def timed_intervals(
    table: pd.DataFrame, prefix: str, start: str, end: str
) -> tuple[NativeAnnotation, ...]:
    """Native intervals with stable ids ``<prefix>#<clean-table row index>``."""
    starts = pd.to_numeric(table[start], errors="coerce")
    ends = pd.to_numeric(table[end], errors="coerce")
    keep = starts.notna() & ends.notna() & (ends > starts)
    return tuple(
        NativeAnnotation(float(s), float(e), f"{prefix}#{index}")
        for index, s, e in zip(table.index[keep], starts[keep], ends[keep], strict=True)
    )


def media_coverage_by_stem(
    media: pd.DataFrame, dataset: str
) -> dict[str, MediaCoverage]:
    """Media-metadata rows of ``dataset`` keyed by file stem (the recording key)."""
    rows = media.loc[media["dataset"].eq(dataset)]
    output: dict[str, MediaCoverage] = {}
    for row in rows.to_dict(orient="records"):
        start = pd.to_numeric(row["audio_start_time_sec"], errors="coerce")
        duration = pd.to_numeric(row["audio_duration_sec"], errors="coerce")
        usable = bool(row["probe_ok"]) and pd.notna(start) and pd.notna(duration)
        output[Path(str(row["relative_path"])).stem] = MediaCoverage(
            probe_ok=usable,
            audio_start_s=float(start) if usable else math.nan,
            audio_end_s=float(start + duration) if usable else math.nan,
        )
    return output


__all__ = [
    "MediaCoverage",
    "NativeFocalVoiceSource",
    "media_coverage_by_stem",
    "require_table",
    "timed_intervals",
]
