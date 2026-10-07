"""Label artifacts on disk: layout, manifests and atomic writes.

```text
${paths.processed}/labels/<dataset>/
  registry.json                 the registry the artifacts were built against
  <extractor>/                  speech | social | text | audio | video
    manifest.json               deterministic: versions, config, inputs, tables
    grid.parquet                (recording_id, decision_index) rows of the action grid
    events.parquet              native-time events        (when the extractor has any)
    segments.parquet            native-time intervals     (when the extractor has any)
    participants.parquet        (recording_id, participant_index) rows
    recordings.parquet          recording_id rows
```

One directory per extractor lets each extractor be rebuilt alone. Consumers
read the Parquet tables directly (column projection, row-group filters on the
keys); `registry.json` names each label's table and columns.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from conv_wm.data.labels.catalog import REGISTRY
from conv_wm.data.labels.registry import (
    Extractor,
    LabelSpec,
    Table,
    registry_document,
)
from conv_wm.data.pipeline_inputs import sha256_file

MANIFEST_FILE = "manifest.json"
REGISTRY_FILE = "registry.json"
ROW_GROUP_SIZE = 65_536
"""Rows per Parquet row group: small enough for recording/time filters to prune."""
COMPRESSION = "zstd"


class StaleLabelsError(ValueError):
    """A label artifact no longer matches what it claims to be built from."""


def table_file(table: Table) -> str:
    """File name of ``table`` inside an extractor directory."""
    return f"{table}.parquet"


def canonical_json(document: Any) -> str:
    """Sorted, indented JSON used for every deterministic document."""
    return json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"


def write_extractor(
    directory: Path, tables: Mapping[Table, pa.Table], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Write one extractor's tables and manifest, replacing ``directory`` atomically.

    Everything is written into a sibling temporary directory first; the old
    directory is swapped out only once every file is complete, so a reader
    never sees a half-written extractor. Returns the manifest as written, with
    each table's checksum, row count and columns filled in.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        entries: dict[str, Any] = {}
        for table, content in sorted(tables.items(), key=lambda item: str(item[0])):
            path = staging / table_file(table)
            pq.write_table(
                content,
                path,
                row_group_size=ROW_GROUP_SIZE,
                compression=COMPRESSION,
            )
            entries[str(table)] = {
                "file": table_file(table),
                "sha256": sha256_file(path),
                "rows": content.num_rows,
                "columns": content.column_names,
            }
        document = {**manifest, "tables": entries}
        (staging / MANIFEST_FILE).write_text(canonical_json(document), encoding="utf-8")
        retired = directory.with_name(f".{directory.name}.retired")
        if retired.exists():
            shutil.rmtree(retired)
        if directory.exists():
            os.replace(directory, retired)
        os.replace(staging, directory)
        if retired.exists():
            shutil.rmtree(retired)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return document


def write_registry(dataset_dir: Path, specs: Sequence[LabelSpec] = REGISTRY) -> Path:
    """Write ``registry.json`` next to a dataset's extractors (atomically)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / REGISTRY_FILE
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(canonical_json(registry_document(specs)), encoding="utf-8")
    os.replace(staging, path)
    return path


def read_manifest(
    dataset_dir: Path, extractor: Extractor | str
) -> dict[str, Any] | None:
    """The manifest of one extractor, or ``None`` when it was never built."""
    path = dataset_dir / str(extractor) / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_extractor(dataset_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Refuse an extractor whose tables no longer match its manifest checksums."""
    extractor = manifest["extractor"]
    for name, entry in manifest.get("tables", {}).items():
        path = dataset_dir / extractor / entry["file"]
        if not path.exists():
            raise StaleLabelsError(f"{path}: listed in its manifest but missing")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise StaleLabelsError(
                f"{path}: checksum {digest} differs from the manifest's "
                f"{entry['sha256']} ({name}); rebuild with conv-wm build labels"
            )


__all__ = [
    "MANIFEST_FILE",
    "REGISTRY_FILE",
    "StaleLabelsError",
    "canonical_json",
    "read_manifest",
    "sha256_file",
    "table_file",
    "verify_extractor",
    "write_extractor",
    "write_registry",
]
