"""Label artifacts on disk: layout, manifests, atomic writes and projected reads.

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

One directory per extractor keeps the number of files small, lets each
extractor be rebuilt alone and guarantees that building audio labels never
touches video ones. Readers only open the manifests and tables of the
extractors that hold a requested label, and read only the requested columns
(Parquet column projection) and rows (row-group filters on the keys).

This module needs nothing but pyarrow: a consumer can read labels without
the build dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from conv_wm.data.labels.catalog import REGISTRY
from conv_wm.data.labels.registry import (
    LABEL_SCHEMA_VERSION,
    PARTICIPANT_AXIS_COLUMN,
    REGISTRY_VERSION,
    TABLE_KEYS,
    Extractor,
    LabelSpec,
    Table,
    registry_document,
)
from conv_wm.data.labels.selection import (
    LabelSelection,
    LabelUnavailableError,
    ResolvedSelection,
    resolve,
)

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


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@dataclass(frozen=True)
class LabelBundle:
    """What :func:`load_labels` returns: one Arrow table per table kind.

    ``tables`` maps ``grid`` / ``events`` / ``segments`` / ``participants`` /
    ``recordings`` to the requested columns (plus the table keys). Grid tables
    of several extractors are joined column-wise on their keys; events and
    segments of several extractors are concatenated (absent columns null).
    """

    tables: Mapping[str, pa.Table] = field(default_factory=dict)
    labels: tuple[str, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict)
    manifests: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __getitem__(self, table: str) -> pa.Table:
        return self.tables[table]

    def __contains__(self, table: object) -> bool:
        return table in self.tables

    def __bool__(self) -> bool:
        return bool(self.tables)


def load_labels(
    dataset_dir: Path,
    selection: LabelSelection,
    *,
    recording_ids: Sequence[str] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    verify: bool = False,
    registry: Sequence[LabelSpec] = REGISTRY,
) -> LabelBundle:
    """Read the labels ``selection`` asks for from one dataset's label store.

    ``selection.enabled == False`` returns an empty bundle before any file is
    touched. Otherwise the selection is resolved against the registry, only
    the manifests of the extractors holding a resolved label are opened, and
    only their columns are read.

    Window semantics, with ``[start_s, end_s)`` on the canonical clock:

    * grid rows whose cell start ``decision_time_s`` lies in the window;
    * events whose ``time_s`` lies in the window (untimed tokens are dropped);
    * segments intersecting the window, returned with their full native bounds;
    * participants / recordings rows of the selected recordings, unwindowed.

    A label named exactly that this store has not materialized raises
    :class:`LabelUnavailableError`; one reached through ``all`` or a wildcard is
    reported in :attr:`LabelBundle.skipped`.
    """
    if not selection.enabled:
        return LabelBundle()
    resolved = resolve(selection, registry)
    explicit = {item for item in selection.include if "*" not in item and item != "all"}
    return _load_resolved(
        dataset_dir,
        resolved,
        explicit=explicit,
        recording_ids=recording_ids,
        start_s=start_s,
        end_s=end_s,
        verify=verify,
    )


def _load_resolved(
    dataset_dir: Path,
    resolved: ResolvedSelection,
    *,
    explicit: set[str],
    recording_ids: Sequence[str] | None,
    start_s: float | None,
    end_s: float | None,
    verify: bool,
) -> LabelBundle:
    skipped = dict(resolved.skipped)
    manifests: dict[str, dict[str, Any]] = {}
    chosen: list[LabelSpec] = []
    for spec in resolved.labels:
        assert spec.extractor is not None
        key = str(spec.extractor)
        if key not in manifests:
            manifest = read_manifest(dataset_dir, spec.extractor)
            if manifest is not None:
                _check_versions(dataset_dir, manifest)
                if verify:
                    verify_extractor(dataset_dir, manifest)
            manifests[key] = manifest or {}
        manifest = manifests[key]
        reason = _unavailable_reason(spec, manifest)
        if reason is None:
            chosen.append(spec)
        elif spec.name in explicit:
            raise LabelUnavailableError(f"{spec.name}: {reason}")
        else:
            skipped[spec.name] = reason
    manifests = {key: value for key, value in manifests.items() if value}
    _check_same_grid(manifests)
    grouped: dict[tuple[str, Table], list[LabelSpec]] = {}
    for spec in chosen:
        assert spec.extractor is not None and spec.table is not None
        grouped.setdefault((str(spec.extractor), spec.table), []).append(spec)
    per_table: dict[Table, list[pa.Table]] = {}
    for (extractor, table), specs in sorted(
        grouped.items(), key=lambda i: (i[0][0], str(i[0][1]))
    ):
        path = dataset_dir / extractor / table_file(table)
        per_table.setdefault(table, []).append(
            _read(path, table, specs, recording_ids, start_s, end_s)
        )
    tables: dict[str, pa.Table] = {}
    for table, parts in per_table.items():
        tables[str(table)] = _combine(table, parts)
    return LabelBundle(
        tables=tables,
        labels=tuple(spec.name for spec in chosen),
        skipped=skipped,
        manifests=manifests,
    )


def _check_versions(dataset_dir: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("label_schema_version") != LABEL_SCHEMA_VERSION:
        raise StaleLabelsError(
            f"{dataset_dir / manifest.get('extractor', '?')}: label schema "
            f"{manifest.get('label_schema_version')} is not {LABEL_SCHEMA_VERSION}; "
            "rebuild with conv-wm build labels"
        )
    if manifest.get("registry_version") != REGISTRY_VERSION:
        raise StaleLabelsError(
            f"{dataset_dir / manifest.get('extractor', '?')}: built with registry "
            f"v{manifest.get('registry_version')}, this code has v{REGISTRY_VERSION}; "
            "rebuild with conv-wm build labels"
        )


def _unavailable_reason(spec: LabelSpec, manifest: Mapping[str, Any]) -> str | None:
    if not manifest:
        return (
            f"extractor {spec.extractor} was not built for this dataset; run "
            f"conv-wm build labels --extractors {spec.extractor}"
        )
    if spec.name in manifest.get("materialized_labels", ()):
        return None
    reasons = manifest.get("unavailable_labels", {})
    return reasons.get(spec.name, "not materialized by this build")


def _check_same_grid(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    digests = {
        name: manifest.get("inputs", {}).get("action_grid", {}).get("sha256")
        for name, manifest in manifests.items()
    }
    if len(set(digests.values())) > 1:
        raise StaleLabelsError(
            f"extractors were built from different action grids {digests}; "
            "rebuild the stale ones with conv-wm build labels"
        )


def _read(
    path: Path,
    table: Table,
    specs: Sequence[LabelSpec],
    recording_ids: Sequence[str] | None,
    start_s: float | None,
    end_s: float | None,
) -> pa.Table:
    columns: list[str] = list(TABLE_KEYS[table])
    if table is Table.GRID and any(spec.participant_axis for spec in specs):
        columns.append(PARTICIPANT_AXIS_COLUMN)
    for spec in specs:
        columns.extend(column for column in spec.columns if column not in columns)
    filters: list[tuple[str, str, Any]] = []
    if recording_ids is not None:
        filters.append(("recording_id", "in", list(recording_ids)))
    row_filters = [spec.row_filter for spec in specs]
    if table in (Table.EVENTS, Table.SEGMENTS) and all(row_filters):
        pairs = [pair for pair in row_filters if pair is not None]
        filters.append((pairs[0][0], "in", sorted({value for _, value in pairs})))
    time_column = {Table.GRID: "decision_time_s", Table.EVENTS: "time_s"}.get(table)
    if time_column is not None:
        if start_s is not None:
            filters.append((time_column, ">=", float(start_s)))
        if end_s is not None:
            filters.append((time_column, "<", float(end_s)))
    elif table is Table.SEGMENTS:
        if end_s is not None:
            filters.append(("start_s", "<", float(end_s)))
        if start_s is not None:
            filters.append(("end_s", ">", float(start_s)))
    return pq.read_table(path, columns=columns, filters=filters or None)


def _combine(table: Table, parts: list[pa.Table]) -> pa.Table:
    if len(parts) == 1:
        return parts[0]
    if table is Table.GRID:
        keys = list(TABLE_KEYS[Table.GRID])
        combined = parts[0]
        for part in parts[1:]:
            for key in keys:
                left, right = part.column(key), combined.column(key)
                if left.type != right.type:  # string vs large_string: compare values
                    left = left.cast(right.type)
                if not left.equals(right):
                    raise StaleLabelsError(
                        "grid tables of two extractors are not aligned on "
                        f"{keys}; rebuild with conv-wm build labels"
                    )
            for name in part.column_names:
                if name not in combined.column_names:
                    combined = combined.append_column(name, part.column(name))
        return combined
    return pa.concat_tables(parts, promote_options="default")


def rows_for(bundle: LabelBundle, spec: LabelSpec) -> pa.Table:
    """The rows and columns of one events/segments label inside a bundle."""
    assert spec.table is not None
    table = bundle.tables[str(spec.table)]
    if spec.row_filter is not None:
        column, value = spec.row_filter
        table = table.filter(pc.field(column) == value)
    return table.select([*TABLE_KEYS[spec.table], *spec.columns])


__all__ = [
    "MANIFEST_FILE",
    "REGISTRY_FILE",
    "LabelBundle",
    "StaleLabelsError",
    "canonical_json",
    "load_labels",
    "read_manifest",
    "rows_for",
    "sha256_file",
    "table_file",
    "verify_extractor",
    "write_extractor",
    "write_registry",
]
