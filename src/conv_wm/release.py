"""Assemble the Hugging Face release directories from the canonical artifacts.

Nothing is recomputed: the action grid and the media manifest are copied
byte for byte, the model-ready index is partitioned by its ``split`` column and
the dataset card is copied with its ``files`` map rewritten to the release
layout. Every input is checked against the checksum the card records.

```text
<output>/<public dataset>/                  one public repository per dataset
    README.md                               hand-written, never overwritten
    metadata.json
    data/action_grid.parquet
    data/media_manifest.parquet
    data/model_ready/<split>.parquet
<output>/full/                              the private combined repository
    README.md                               hand-written, never overwritten
    <dataset>/metadata.json
    <dataset>/action_grid.parquet
    <dataset>/media_manifest.parquet
    <dataset>/model_ready/<split>.parquet
```

Raw media is never copied: the manifest only references it.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.pipeline.action_grid import GRID_TABLE, PROCESSED_ROOT
from conv_wm.data.pipeline.model_ready import (
    CANONICAL_SPLITS,
    INDEX_TABLE,
    METADATA_FILE,
)
from conv_wm.data.pipeline.model_ready import REPORT_FILE as MODEL_READY_REPORT
from conv_wm.data.pipeline.model_ready import REPORT_ROOT as MODEL_READY_ROOT
from conv_wm.data.pipeline_inputs import sha256_file
from conv_wm.reports import JsonDict

FULL_RELEASE = "full"
GRID_FILE = "action_grid.parquet"
MEDIA_FILE = "media_manifest.parquet"
PARTITION_DIR = "model_ready"


@dataclass(frozen=True)
class ReleaseDataset:
    """The files one dataset contributes to one release directory."""

    dataset: str
    metadata_path: Path
    data_dir: Path
    splits: tuple[str, ...]


def _require_checksum(path: Path, expected: str | None, dataset: str) -> Path:
    if not path.exists():
        raise MissingPrerequisiteError(
            path, produce_with=f"conv-wm build all --dataset {dataset}"
        )
    if expected is None or sha256_file(path) != expected:
        raise ValueError(
            f"{path} does not match the checksum its dataset card records; "
            f"rerun conv-wm build all --dataset {dataset}"
        )
    return path


def write_release_dataset(
    cfg: DictConfig, dataset: str, *, metadata_path: Path, data_dir: Path
) -> ReleaseDataset:
    """Copy one dataset's canonical artifacts into a release layout."""
    paths = pipeline_paths(cfg)
    source_dir = paths.model_ready / dataset
    card_path = source_dir / METADATA_FILE
    if not card_path.exists():
        raise MissingPrerequisiteError(
            card_path, produce_with=f"conv-wm build model-ready --dataset {dataset}"
        )
    card: JsonDict = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("media") is None:
        raise MissingPrerequisiteError(
            source_dir / MEDIA_FILE,
            produce_with=f"conv-wm build all --dataset {dataset} --from media-manifest",
        )
    report_path = paths.reports / MODEL_READY_ROOT / dataset / MODEL_READY_REPORT
    report: JsonDict = json.loads(report_path.read_text(encoding="utf-8"))

    grid = _require_checksum(
        paths.processed / PROCESSED_ROOT / dataset / GRID_TABLE,
        card["source"]["vocal_action_grid"]["sha256"],
        dataset,
    )
    media = _require_checksum(source_dir / MEDIA_FILE, card["media"]["sha256"], dataset)
    index_path = _require_checksum(
        source_dir / INDEX_TABLE,
        report["output_artifacts"]["index"]["sha256"],
        dataset,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(grid, data_dir / GRID_FILE)
    shutil.copyfile(media, data_dir / MEDIA_FILE)
    index = pd.read_parquet(index_path)
    partitions = data_dir / PARTITION_DIR
    if partitions.exists():
        shutil.rmtree(partitions)
    partitions.mkdir()
    splits = tuple(
        split for split in CANONICAL_SPLITS if index["split"].eq(split).any()
    )
    for split in splits:
        index.loc[index["split"].eq(split)].reset_index(drop=True).to_parquet(
            partitions / f"{split}.parquet", index=False
        )

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, metadata_path.parent)).as_posix()

    card["files"] = {
        "index": {split: relative(partitions / f"{split}.parquet") for split in splits},
        "sequences": relative(data_dir / GRID_FILE),
        "media_manifest": relative(data_dir / MEDIA_FILE),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ReleaseDataset(dataset, metadata_path, data_dir, splits)


def build_hf_release(
    cfg: DictConfig, *, output: Path | None = None
) -> list[ReleaseDataset]:
    """Write the public per-dataset releases and the private combined one."""
    root = Path(output or cfg.release.output)
    written: list[ReleaseDataset] = []
    for dataset in cfg.release.public:
        base = root / str(dataset)
        written.append(
            write_release_dataset(
                cfg,
                str(dataset),
                metadata_path=base / METADATA_FILE,
                data_dir=base / "data",
            )
        )
    for dataset in cfg.release.full:
        base = root / FULL_RELEASE / str(dataset)
        written.append(
            write_release_dataset(
                cfg, str(dataset), metadata_path=base / METADATA_FILE, data_dir=base
            )
        )
    return written


__all__ = ["ReleaseDataset", "build_hf_release", "write_release_dataset"]
