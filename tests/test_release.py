"""The Hugging Face release layout, assembled from checked canonical artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "data"))

from state_layers import (
    build_action_grid,
    config_for,
    long_timeline,
    write_media_metadata,
    write_native_state,
    write_recording_splits,
)

from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.pipeline.media_manifest import run_media_manifest_build
from conv_wm.data.pipeline.model_ready import run_model_ready_build
from conv_wm.release import build_hf_release


def _built(tmp_path, *, media=True):
    cfg = config_for(tmp_path)
    cfg.release = OmegaConf.create(
        {"output": str(tmp_path / "release"), "public": ["egocom"], "full": ["egocom"]}
    )
    write_native_state(cfg, "egocom", long_timeline(), recording="rec-1")
    build_action_grid(cfg, "egocom")
    write_recording_splits(cfg, "egocom", {"rec-1": "train"})
    write_media_metadata(cfg, [("egocom", "EgoCom/240p/rec-1.MP4", 1)])
    if media:
        run_media_manifest_build(cfg, dataset="egocom")
    run_model_ready_build(cfg, dataset="egocom")
    return cfg


def test_release_copies_the_canonical_artifacts_into_both_layouts(tmp_path):
    cfg = _built(tmp_path)

    build_hf_release(cfg)

    root = tmp_path / "release"
    public = json.loads((root / "egocom" / "metadata.json").read_text())
    full = json.loads((root / "full" / "egocom" / "metadata.json").read_text())
    assert public["files"] == {
        "index": {"train": "data/model_ready/train.parquet"},
        "sequences": "data/action_grid.parquet",
        "media_manifest": "data/media_manifest.parquet",
    }
    assert full["files"]["media_manifest"] == "media_manifest.parquet"
    grid = (
        Path(cfg.paths.processed) / "vocal_action_grid/egocom/vocal_action_grid.parquet"
    )
    assert (root / "egocom/data/action_grid.parquet").read_bytes() == grid.read_bytes()
    index = pd.read_parquet(Path(cfg.paths.model_ready) / "egocom" / "index.parquet")
    train = pd.read_parquet(root / "egocom/data/model_ready/train.parquet")
    assert train.equals(index.loc[index["split"].eq("train")].reset_index(drop=True))
    media = pd.read_parquet(root / "full/egocom/media_manifest.parquet")
    assert media["video_path"].tolist() == ["240p/rec-1.MP4"]
    assert str(tmp_path) not in json.dumps(public)
    assert not list(root.rglob("*.MP4"))


def test_release_refuses_a_dataset_without_a_media_manifest(tmp_path):
    cfg = _built(tmp_path, media=False)

    with pytest.raises(MissingPrerequisiteError, match="media-manifest"):
        build_hf_release(cfg)


def test_release_refuses_an_artifact_changed_behind_its_card(tmp_path):
    cfg = _built(tmp_path)
    media = Path(cfg.paths.model_ready) / "egocom" / "media_manifest.parquet"
    media.write_bytes(media.read_bytes() + b"\0")

    with pytest.raises(ValueError, match="checksum"):
        build_hf_release(cfg)
