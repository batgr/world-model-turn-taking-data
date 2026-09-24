"""Writing generated report artifacts with stable, provenance-stamped layouts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conv_wm.provenance import ReportProvenance, collect_provenance

SUMMARY_SCHEMA_VERSION = 1
"""Bumped when the top-level layout of summary files changes."""

JsonDict = dict[str, Any]
"""A JSON-serializable mapping; the honest type of nested report documents."""


def write_summary(
    path: Path,
    payload: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    provenance: ReportProvenance | None = None,
) -> JsonDict:
    """Write ``payload`` as a JSON summary stamped with execution provenance.

    The written document is ``payload`` plus three reserved keys:
    ``summary_schema_version``, ``parameters`` (audit settings that materially
    affect interpretation, when given) and ``provenance``. Keys are sorted so the
    file is diff-friendly. Returns the written document.
    """
    reserved = {"provenance", "parameters", "summary_schema_version"} & set(payload)
    if reserved:
        raise ValueError(f"payload must not define reserved keys: {sorted(reserved)}")
    document: JsonDict = {
        **payload,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "provenance": (provenance or collect_provenance()).to_dict(),
    }
    if parameters is not None:
        document["parameters"] = dict(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(path)
    staging.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)
    return document


def write_table(path: Path, table: pd.DataFrame) -> Path:
    """Write ``table`` as Parquet without the index, creating parent directories.

    The file is written next to its destination and renamed into place, so a
    reader never sees a partially written table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(path)
    try:
        table.to_parquet(staging, index=False)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return path


def _staging_path(path: Path) -> Path:
    """A same-directory temporary name, so the final rename is atomic."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _json_default(value: object) -> object:
    """Serialize numpy scalars and paths that ``json`` does not handle natively."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
