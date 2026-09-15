"""Writing generated report artifacts with stable, provenance-stamped layouts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from conv_wm.provenance import ReportProvenance, collect_provenance

SUMMARY_SCHEMA_VERSION = 1
"""Bumped when the top-level layout of summary files changes."""


def write_summary(
    path: Path,
    payload: Mapping[str, object],
    *,
    parameters: Mapping[str, object] | None = None,
    provenance: ReportProvenance | None = None,
) -> dict[str, object]:
    """Write ``payload`` as a JSON summary stamped with execution provenance.

    The written document is ``payload`` plus three reserved keys:
    ``summary_schema_version``, ``parameters`` (audit settings that materially
    affect interpretation, when given) and ``provenance``. Keys are sorted so the
    file is diff-friendly. Returns the written document.
    """
    reserved = {"provenance", "parameters", "summary_schema_version"} & set(payload)
    if reserved:
        raise ValueError(f"payload must not define reserved keys: {sorted(reserved)}")
    document: dict[str, object] = {
        **payload,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "provenance": (provenance or collect_provenance()).to_dict(),
    }
    if parameters is not None:
        document["parameters"] = dict(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return document


def write_table(path: Path, table: pd.DataFrame) -> Path:
    """Write ``table`` as Parquet without the index, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def _json_default(value: object) -> object:
    """Serialize numpy scalars and paths that ``json`` does not handle natively."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
