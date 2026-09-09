from pathlib import Path
from collections.abc import Iterator, Mapping
import hashlib
import pandas as pd

FILE_TYPE_BY_EXTENSION = {
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".avi": "video",
    ".wav": "audio",
    ".flac": "audio",
    ".mp3": "audio",
    ".m4a": "audio",
    ".csv": "tabular",
    ".parquet": "tabular",
    ".json": "structured_data",
}

IGNORED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

IGNORED_DIRS = {
    "__MACOSX",
}
MANIFEST_COLUMNS = [
    "dataset",
    "relative_path",
    "file_name",
    "extension",
    "file_type",
    "size_bytes",
    "checksum",
    "checksum_algorithm",
]

MANIFEST_DTYPES = {
    "dataset": "string",
    "relative_path": "string",
    "file_name": "string",
    "extension": "string",
    "file_type": "string",
    "size_bytes": "uint64",
    "checksum": "string",
    "checksum_algorithm": "string",
}


def build_manifest(
    raw_root: Path,
    *,
    compute_checksum: bool = False,
) -> pd.DataFrame:
    """Build a deterministic manifest of all files under the raw data root.

    The first directory below ``raw_root`` is interpreted as the dataset name.

    Example:
        raw/EgoCom/... -> dataset="egocom"
        raw/Ego4D/...  -> dataset="ego4d"

    Args:
        raw_root: Root directory containing all raw datasets.
        compute_checksum: Whether to compute SHA-256 checksums for each file.

    Returns:
        A deterministic manifest with one row per discovered file.

    Raises:
        FileNotFoundError: If ``raw_root`` does not exist.
        NotADirectoryError: If ``raw_root`` is not a directory.
    """
    raw_root = Path(raw_root)
    records = []

    for path in _iter_raw_files(raw_root):
        relative_path = path.relative_to(raw_root)

        dataset = (
            relative_path.parts[0].lower()
            if len(relative_path.parts) > 1
            else "unknown"
        )

        records.append(
            {
                "dataset": dataset,
                "relative_path": relative_path.as_posix(),
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "file_type": _classify_file_type(path),
                "size_bytes": path.stat().st_size,
                "checksum": (_compute_sha256(path) if compute_checksum else None),
                "checksum_algorithm": ("sha256" if compute_checksum else None),
            }
        )

    manifest = pd.DataFrame.from_records(
        records,
        columns=MANIFEST_COLUMNS,
    ).astype(MANIFEST_DTYPES)

    validate_manifest(manifest)
    return manifest.sort_values(
        ["dataset", "relative_path"],
        ignore_index=True,
    )


def save_manifest(
    manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist a manifest as a Parquet file.

    Args:
        manifest: Manifest dataframe to save.
        output_path: Destination Parquet path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)


def _classify_file_type(path: Path) -> str:
    """Infer a broad file type from its extension.

    Args:
        path: File path to classify.

    Returns:
        One of ``"video"``, ``"audio"``, ``"annotation"``, or ``"other"``.
    """
    return FILE_TYPE_BY_EXTENSION.get(path.suffix.lower(), "other")


def _iter_raw_files(root: Path) -> Iterator[Path]:
    """Yield raw files recursively in deterministic order.

    Args:
        root: Root directory containing the raw datasets.

    Yields:
        Files under ``root`` excluding known system artifacts,
        sorted deterministically by relative path.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"Raw data root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Raw data root is not a directory: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in IGNORED_NAMES
        and not path.name.startswith("._")
        and not any(part in IGNORED_DIRS for part in path.parts)
    ]

    yield from sorted(
        files,
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        path: File to hash.
        chunk_size: Number of bytes read per iteration.

    Returns:
        Hexadecimal SHA-256 digest of the file.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Validate core manifest invariants."""
    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    if manifest.duplicated(["dataset", "relative_path"]).any():
        raise ValueError("Manifest contains duplicate dataset/relative_path entries.")

    if (manifest["size_bytes"] < 0).any():
        raise ValueError("Manifest contains negative file sizes.")

    if manifest["relative_path"].str.strip().eq("").any():
        raise ValueError("Manifest contains empty relative paths.")
