"""Raw dataset inventory: build and persist the manifest of the raw root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.data.manifest import build_manifest, save_manifest
from conv_wm.reports import write_summary


@dataclass(frozen=True)
class ManifestOutputs:
    """Artifacts written by one run."""

    manifest: pd.DataFrame
    manifest_path: Path
    summary: dict[str, object]

    @property
    def summary_path(self) -> Path:
        """Location of the inventory summary."""
        return self.manifest_path.with_name("summary.json")


def build_manifest_summary(manifest: pd.DataFrame) -> dict[str, object]:
    """File counts and sizes by dataset and file type."""
    return {
        "audit_type": "raw_dataset_inventory",
        "n_files": len(manifest),
        "total_size_bytes": int(manifest["size_bytes"].sum()),
        "checksums_computed": bool(manifest["checksum"].notna().all())
        if len(manifest)
        else False,
        "datasets": {
            str(dataset): {
                "n_files": len(group),
                "total_size_bytes": int(group["size_bytes"].sum()),
                "file_types": {
                    str(kind): int(count)
                    for kind, count in group["file_type"].value_counts().items()
                },
            }
            for dataset, group in manifest.groupby("dataset")
        },
    }


def run_manifest_audit(cfg: DictConfig) -> ManifestOutputs:
    """Inventory the configured raw root and write manifest and summary."""
    manifest = build_manifest(
        Path(cfg.paths.raw), compute_checksum=bool(cfg.manifest.compute_checksum)
    )
    manifest_path = Path(cfg.manifest.output)
    save_manifest(manifest, manifest_path)
    summary = build_manifest_summary(manifest)
    outputs = ManifestOutputs(manifest, manifest_path, summary)
    write_summary(
        outputs.summary_path,
        summary,
        parameters={"compute_checksum": bool(cfg.manifest.compute_checksum)},
    )
    return outputs
