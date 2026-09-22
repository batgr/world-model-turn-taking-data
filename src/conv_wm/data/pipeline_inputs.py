"""Reading one pipeline layer's artifact as the validated input of the next.

Every derived layer writes a table plus a report that records the table's
schema version and checksum. A downstream builder must refuse an input whose
report no longer describes the bytes on disk: a stale or partially rewritten
layer is a hard error, never a silently consumed input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.reports import JsonDict


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_reference(path: Path) -> dict[str, str]:
    """The ``{path, sha256}`` pair every report records for an artifact."""
    return {"path": str(path), "sha256": sha256_file(path)}


@dataclass(frozen=True)
class CheckedArtifact:
    """One upstream layer's table, its report, and the checksums that tie them."""

    dataset: str
    table: pd.DataFrame
    table_path: Path
    table_sha256: str
    report_path: Path
    report_sha256: str
    report: JsonDict


def load_checked_artifact(
    *,
    dataset: str,
    table_path: Path,
    report_path: Path,
    schema_key: str,
    expected_version: int,
    artifact_key: str,
    produce_with: str,
) -> CheckedArtifact:
    """Read ``table_path`` only if ``report_path`` still vouches for it.

    The report's ``schema_key`` must be the version the caller understands and
    its recorded ``output_artifacts[artifact_key].sha256`` must equal the table
    on disk; otherwise the build stops and names the command to rerun.
    """
    for path in (table_path, report_path):
        if not path.exists():
            raise MissingPrerequisiteError(path, produce_with=produce_with)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    version = report.get(schema_key)
    if version != expected_version:
        raise ValueError(
            f"{report_path}: {schema_key} {version} is not the expected "
            f"{expected_version}; rerun {produce_with}"
        )
    recorded = report.get("output_artifacts", {}).get(artifact_key, {}).get("sha256")
    table_sha256 = sha256_file(table_path)
    if recorded != table_sha256:
        raise ValueError(
            f"{table_path}: checksum {table_sha256} does not match the one recorded "
            f"in {report_path} ({recorded}); rerun {produce_with}"
        )
    return CheckedArtifact(
        dataset=dataset,
        table=pd.read_parquet(table_path),
        table_path=table_path,
        table_sha256=table_sha256,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        report=report,
    )


__all__ = [
    "CheckedArtifact",
    "artifact_reference",
    "load_checked_artifact",
    "sha256_file",
]
