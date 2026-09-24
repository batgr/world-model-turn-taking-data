"""The canonical media manifest: ``(dataset, recording_id)`` -> raw media files.

One row per canonical recording whose media could be resolved. Paths are
relative to the dataset's *corpus root* — its directory below the raw root
(``raw/EgoCom``, ``raw/Ego4D``) — so the artifact carries no machine-specific
location: a consumer joins its own configured root with the stored path.

Adapters describe media with :class:`MediaRecord` rows whose paths are the
media metadata audit's ``relative_path`` (relative to the raw root); the
generic code here strips the corpus directory, validates the contract and
measures coverage. Nothing in this module branches on a dataset name.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from conv_wm.data.manifest import FILE_TYPE_BY_EXTENSION

MEDIA_SCHEMA_VERSION = 1
"""Bumped when a column, its meaning or the path convention changes."""

MEDIA_MANIFEST_COLUMNS = (
    "dataset",
    "recording_id",
    "video_path",
    "audio_path",
    "media_offset_s",
    "video_has_audio",
)
MEDIA_MANIFEST_DTYPES = {
    "dataset": "string",
    "recording_id": "string",
    "video_path": "string",
    "audio_path": "string",
    "media_offset_s": "float64",
    "video_has_audio": "boolean",
}
COLUMN_SEMANTICS = {
    "dataset": "corpus key, as in every canonical artifact",
    "recording_id": "canonical recording id, identical to the action grid's",
    "video_path": "video file relative to the corpus root, or null",
    "audio_path": "separate audio file relative to the corpus root, or null",
    "media_offset_s": (
        "media_time_s = decision_time_s + media_offset_s: where the recording's "
        "canonical clock sits on the media file's own timeline"
    ),
    "video_has_audio": (
        "whether the video container carries an audio stream (null when the "
        "media audit could not probe it); audio_path null with video_has_audio "
        "true means the audio is decoded from the video container"
    ),
}


class MediaManifestError(ValueError):
    """The media manifest violates its contract; the message lists every violation."""


class AmbiguousMediaError(MediaManifestError):
    """More than one media file of a kind matches one recording."""


@dataclass(frozen=True)
class MediaFile:
    """One file of the media metadata audit, as an adapter picks it."""

    relative_path: str
    """Path relative to the raw root (the audit's ``relative_path``)."""
    has_audio_stream: bool | None
    """Whether the probe found an audio stream; ``None`` when it could not probe."""


@dataclass(frozen=True)
class MediaRecord:
    """The media of one canonical recording, as an adapter resolves it."""

    recording_id: str
    video: MediaFile | None
    audio: MediaFile | None
    media_offset_s: float = 0.0
    """Position of the recording's canonical time 0 on the media timeline."""


@dataclass(frozen=True)
class MediaSource:
    """What a dataset adapter returns for the requested canonical recordings."""

    dataset: str
    records: tuple[MediaRecord, ...]
    unresolved: Mapping[str, str] = field(default_factory=dict)
    """``recording_id -> reason`` for every requested recording without media."""
    input_paths: tuple[Path, ...] = ()
    """Adapter tables the mapping was derived from (checksummed in the report)."""
    mapping: str = ""
    """One sentence: how the adapter maps a recording to its files and clock."""


class MediaFileIndex:
    """Media metadata rows of one dataset, keyed by file stem and kind.

    Kind comes from the extension (``video`` / ``audio``), exactly as the raw
    manifest classifies it. ``resolve`` refuses a key that matches several
    files of the same kind instead of picking one.
    """

    def __init__(self, media: pd.DataFrame, dataset: str) -> None:
        self._files: dict[tuple[str, str], list[MediaFile]] = {}
        rows = media.loc[media["dataset"].eq(dataset)]
        for row in rows.to_dict(orient="records"):
            path = PurePosixPath(str(row["relative_path"]))
            kind = FILE_TYPE_BY_EXTENSION.get(path.suffix.lower())
            if kind not in ("video", "audio"):
                continue
            self._files.setdefault((path.stem, kind), []).append(
                MediaFile(str(path), _has_audio_stream(row))
            )

    def resolve(self, key: str) -> tuple[MediaFile | None, MediaFile | None]:
        """The unique ``(video, audio)`` files whose stem is ``key``; ``None`` if absent."""
        return self._unique(key, "video"), self._unique(key, "audio")

    def _unique(self, key: str, kind: str) -> MediaFile | None:
        files = self._files.get((key, kind), [])
        if len(files) > 1:
            paths = sorted(file.relative_path for file in files)
            raise AmbiguousMediaError(f"{key!r} matches several {kind} files: {paths}")
        return files[0] if files else None


def _has_audio_stream(row: Mapping[Hashable, Any]) -> bool | None:
    if not bool(row.get("probe_ok", False)):
        return None
    count = row.get("n_audio_streams")
    return None if count is None or pd.isna(count) else bool(float(count) >= 1)


def split_corpus_path(dataset: str, raw_relative_path: str) -> tuple[str, str]:
    """``"EgoCom/240p/x.MP4"`` -> ``("EgoCom", "240p/x.MP4")`` for dataset ``egocom``.

    The raw manifest names a dataset after its first directory below the raw
    root, lower-cased; that directory is the corpus root.
    """
    parts = PurePosixPath(raw_relative_path).parts
    if len(parts) < 2 or parts[0].lower() != dataset:
        raise MediaManifestError(
            f"{raw_relative_path!r} is not below the {dataset!r} corpus directory"
        )
    return parts[0], PurePosixPath(*parts[1:]).as_posix()


def media_manifest_table(source: MediaSource) -> tuple[pd.DataFrame, str | None]:
    """The canonical table of ``source`` and the corpus directory its paths are under."""
    directories: set[str] = set()
    rows: list[dict[str, object]] = []

    def relative(file: MediaFile | None) -> str | None:
        if file is None:
            return None
        directory, path = split_corpus_path(source.dataset, file.relative_path)
        directories.add(directory)
        return path

    for record in sorted(source.records, key=lambda item: item.recording_id):
        rows.append(
            {
                "dataset": source.dataset,
                "recording_id": record.recording_id,
                "video_path": relative(record.video),
                "audio_path": relative(record.audio),
                "media_offset_s": float(record.media_offset_s),
                "video_has_audio": (
                    record.video.has_audio_stream if record.video else None
                ),
            }
        )
    if len(directories) > 1:
        raise MediaManifestError(
            f"{source.dataset}: media spans several corpus directories {sorted(directories)}"
        )
    table = pd.DataFrame.from_records(rows, columns=list(MEDIA_MANIFEST_COLUMNS))
    return table.astype(MEDIA_MANIFEST_DTYPES), next(iter(directories), None)


def is_portable_path(path: str) -> bool:
    """A non-empty, relative POSIX path that never climbs above its root."""
    if not path or path != path.strip() or "\\" in path:
        return False
    if path.startswith("~") or (len(path) > 1 and path[1] == ":"):
        return False
    pure = PurePosixPath(path)
    return not pure.is_absolute() and ".." not in pure.parts


def validate_media_manifest(
    table: pd.DataFrame, canonical: Iterable[tuple[str, str]]
) -> None:
    """Raise :class:`MediaManifestError` listing every violated invariant.

    ``canonical`` is the set of ``(dataset, recording_id)`` pairs the generated
    data knows (the action grid's); a row outside it is rejected.
    """
    missing = set(MEDIA_MANIFEST_COLUMNS) - set(table.columns)
    if missing:
        raise MediaManifestError(f"media manifest lacks columns {sorted(missing)}")
    problems: list[str] = []
    for column in ("dataset", "recording_id"):
        empty = table[column].isna() | table[column].astype("string").str.strip().eq("")
        if empty.any():
            problems.append(f"{int(empty.sum())} rows with a null {column}")
    no_media = table["video_path"].isna() & table["audio_path"].isna()
    if no_media.any():
        problems.append(
            f"{int(no_media.sum())} rows with neither video_path nor audio_path: "
            f"{_examples(table.loc[no_media, 'recording_id'])}"
        )
    duplicated = table.duplicated(["dataset", "recording_id"], keep=False)
    if duplicated.any():
        problems.append(
            "ambiguous (dataset, recording_id) mappings: "
            f"{_examples(table.loc[duplicated, 'recording_id'])}"
        )
    for column in ("video_path", "audio_path"):
        paths = table[column].dropna().astype(str)
        bad = paths.loc[~paths.map(is_portable_path)]
        if not bad.empty:
            problems.append(f"non-portable {column} values: {_examples(bad)}")
    offsets = pd.to_numeric(table["media_offset_s"], errors="coerce")
    if offsets.isna().any():
        problems.append(f"{int(offsets.isna().sum())} rows with a null media_offset_s")
    known = set(canonical)
    keys = zip(table["dataset"].astype(str), table["recording_id"].astype(str))
    unknown = [
        f"{dataset}/{rid}" for dataset, rid in keys if (dataset, rid) not in known
    ]
    if unknown:
        problems.append(f"recordings unknown to the canonical data: {unknown[:5]}")
    if problems:
        raise MediaManifestError("; ".join(problems))


def _examples(values: pd.Series, limit: int = 5) -> list[str]:
    return sorted({str(value) for value in values})[:limit]


def media_coverage(
    table: pd.DataFrame,
    canonical_recordings: Iterable[str],
    unresolved: Mapping[str, str],
    valid_slots: Mapping[str, int],
) -> dict[str, object]:
    """Counts of one dataset's media coverage and the recordings left unresolved.

    ``valid_slots`` maps a canonical recording to its number of labelled
    (``action_valid``) grid slots: an unresolved recording that still carries
    labels is *unexpected* — the grid claims evidence the media cannot back.
    """
    canonical = sorted(set(canonical_recordings))
    video = table["video_path"].notna()
    audio = table["audio_path"].notna()
    resolved = set(table["recording_id"].astype(str))
    missing = [rid for rid in canonical if rid not in resolved]
    unexpected = [rid for rid in missing if valid_slots.get(rid, 0) > 0]
    reasons: dict[str, int] = {}
    for rid in missing:
        reason = unresolved.get(rid, "not_returned_by_adapter")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "recordings_in_action_grid": len(canonical),
        "recordings_in_media_manifest": len(table),
        "video_available": int(video.sum()),
        "audio_path_available": int(audio.sum()),
        "video_only": int((video & ~audio).sum()),
        "audio_only": int((~video & audio).sum()),
        "both": int((video & audio).sum()),
        "video_with_embedded_audio": int(table["video_has_audio"].eq(True).sum()),
        "recordings_without_usable_audio": int(
            (~audio & ~table["video_has_audio"].eq(True)).sum()
        ),
        "recordings_without_resolved_media": len(missing),
        "unresolved_reasons": dict(sorted(reasons.items())),
        "unresolved_recordings": {rid: unresolved.get(rid, "") for rid in missing},
        "unresolved_with_valid_slots": unexpected,
    }


__all__ = [
    "COLUMN_SEMANTICS",
    "MEDIA_MANIFEST_COLUMNS",
    "MEDIA_MANIFEST_DTYPES",
    "MEDIA_SCHEMA_VERSION",
    "AmbiguousMediaError",
    "MediaFile",
    "MediaFileIndex",
    "MediaManifestError",
    "MediaRecord",
    "MediaSource",
    "is_portable_path",
    "media_coverage",
    "media_manifest_table",
    "split_corpus_path",
    "validate_media_manifest",
]
