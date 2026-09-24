"""Smoke tests of the label build on real corpus annotations (opt-in).

Set the path of a local copy to run them; otherwise they are skipped:

* ``CONV_WM_EGOCOM_ANNOTATIONS`` — directory with EgoCom's ``video_info.csv``
  and ``ground_truth_transcriptions.csv``; ``CONV_WM_EGOCOM_PARTS`` limits the
  run to the first N conversation parts (default 6);
* ``CONV_WM_EGO4D_ANNOTATIONS`` — directory with Ego4D's v2 ``av_train.json``
  and ``av_val.json``.

Media files are not needed: the media metadata audit table is synthesized as
if every recording's audio covered its whole annotated window, so these tests
exercise the annotations, never the media. They run the real cleaning and the
real ``conv-wm build all``, then check the label contract on real data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from omegaconf import DictConfig, OmegaConf

from conv_wm import cli
from conv_wm.config import load_config
from conv_wm.data.labels import store
from conv_wm.data.labels.selection import LabelSelection
from conv_wm.data.pipeline.clean import run_annotation_cleaning
from conv_wm.data.pipeline.labels import dataset_dir

pytestmark = pytest.mark.integration

EGOCOM = os.environ.get("CONV_WM_EGOCOM_ANNOTATIONS")
EGO4D = os.environ.get("CONV_WM_EGO4D_ANNOTATIONS")


def _config(tmp_path: Path):
    cfg = load_config()
    cfg.root = str(tmp_path)
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(resolved, DictConfig)
    path = tmp_path / "config.yaml"
    path.write_text(OmegaConf.to_yaml(resolved))
    manifest = Path(resolved.manifest.output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"relative_path": ["x"]}).to_parquet(manifest, index=False)
    return resolved, path


def _media(cfg, rows: list[dict]) -> None:
    path = (
        Path(cfg.paths.reports)
        / "temporal"
        / "media_metadata"
        / "media_metadata.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _check_label_contract(cfg, dataset: str) -> dict:
    root = dataset_dir(cfg, dataset)
    speech = store.read_manifest(root, "speech")
    assert speech is not None
    store.verify_extractor(root, speech)
    report = json.loads(
        (
            Path(cfg.paths.reports) / "labels" / dataset / "speech" / "report.json"
        ).read_text()
    )
    assert report["native_focal_voice_state_agreement"]["recordings_mismatched"] == 0
    grid = pq.read_table(
        root / "speech" / "grid.parquet", columns=["recording_id", "decision_index"]
    )
    keys = grid.to_pandas()
    assert not keys.duplicated().any()
    action = pd.read_parquet(
        Path(cfg.paths.processed)
        / "vocal_action_grid"
        / dataset
        / "vocal_action_grid.parquet",
        columns=["recording_id", "decision_index"],
    )
    assert len(action) == len(keys)
    segments = pq.read_table(root / "speech" / "segments.parquet").to_pandas()
    assert (segments.end_s > segments.start_s).all()
    events = pq.read_table(root / "speech" / "events.parquet").to_pandas()
    assert (
        events.groupby("recording_id")
        .time_s.apply(lambda t: t.is_monotonic_increasing)
        .all()
    )
    bundle = store.load_labels(
        root,
        LabelSelection(
            True, ("timing.time_to_next_ego_onset", "instantaneous.ego_speaking")
        ),
    )
    values = (
        bundle["grid"].column("time_to_next_ego_onset").to_numpy(zero_copy_only=False)
    )
    valid = (
        bundle["grid"]
        .column("time_to_next_ego_onset_valid")
        .to_numpy(zero_copy_only=False)
    )
    assert np.isnan(values.astype(float)[~valid]).all()  # censored stays null, never 0
    assert (values.astype(float)[valid] >= 0).all()
    assert (
        cli.main(
            [
                "--config",
                str(Path(cfg.root) / "config.yaml"),
                "audit",
                "labels",
                "--dataset",
                dataset,
            ]
        )
        == 0
    )
    coverage = json.loads(
        (
            Path(cfg.paths.reports) / "labels" / "coverage" / f"coverage_{dataset}.json"
        ).read_text()
    )
    return coverage["datasets"][dataset]


@pytest.mark.skipif(EGOCOM is None, reason="set CONV_WM_EGOCOM_ANNOTATIONS")
def test_egocom_labels_on_real_annotations(tmp_path):
    assert EGOCOM is not None
    cfg, config_path = _config(tmp_path)
    source = Path(EGOCOM)
    info = pd.read_csv(source / "video_info.csv")
    parts = list(dict.fromkeys(info.conversation_id))[
        : int(os.environ.get("CONV_WM_EGOCOM_PARTS", "6"))
    ]
    info = info[info.conversation_id.isin(parts)]
    words = pd.read_csv(source / "ground_truth_transcriptions.csv")
    words = words[words.conversation_id.isin(parts)]
    raw = Path(cfg.datasets.egocom.raw)
    raw.mkdir(parents=True, exist_ok=True)
    info.to_csv(raw / "video_info.csv", index=False)
    words.to_csv(raw / "ground_truth_transcriptions.csv", index=False)
    run_annotation_cleaning(cfg, dataset_names=["egocom"])
    _media(
        cfg,
        [
            {
                "dataset": "egocom",
                "relative_path": f"EgoCom/{name}.MP4",
                "probe_ok": True,
                "n_audio_streams": 1,
                "audio_start_time_sec": 0.0,
                "audio_duration_sec": float(duration) + 1.0,
            }
            for name, duration in zip(
                info.video_name, info.duration_seconds, strict=True
            )
        ],
    )
    assert (
        cli.main(["--config", str(config_path), "build", "all", "--dataset", "egocom"])
        == 0
    )
    coverage = _check_label_contract(cfg, "egocom")
    labels = coverage["labels"]
    assert labels["instantaneous.ego_speaking"]["valid"] == pytest.approx(1.0)
    assert labels["text.speech_rate"]["materialized"]
    assert not labels["social_native.looking_at_wearer_subframes"]["supported"]
    sanity = coverage["sanity"]
    assert sanity["floor_transfer_count"] > 0 and sanity["ego_onset_count"] > 0
    assert 0.0 < sanity["overlap_share_of_known_subframes"] < 0.5
    recordings = pq.read_table(
        dataset_dir(cfg, "egocom") / "speech" / "recordings.parquet"
    )
    counts = recordings.column("participant_count").to_pylist()
    assert min(counts) >= 2


@pytest.mark.skipif(EGO4D is None, reason="set CONV_WM_EGO4D_ANNOTATIONS")
def test_ego4d_labels_on_real_annotations(tmp_path):
    assert EGO4D is not None
    cfg, config_path = _config(tmp_path)
    cfg.datasets.ego4d.raw = EGO4D
    config_path.write_text(OmegaConf.to_yaml(cfg))
    run_annotation_cleaning(cfg, dataset_names=["ego4d"])
    clips = pd.read_parquet(Path(cfg.datasets.ego4d.interim) / "clips_clean.parquet")
    _media(
        cfg,
        [
            {
                "dataset": "ego4d",
                "relative_path": f"Ego4D/v2/full_scale/{uid}.mp4",
                "probe_ok": True,
                "n_audio_streams": 1,
                "audio_start_time_sec": 0.0,
                "audio_duration_sec": 1e6,
            }
            for uid in sorted(set(clips.video_uid))
        ],
    )
    assert (
        cli.main(["--config", str(config_path), "build", "all", "--dataset", "ego4d"])
        == 0
    )
    coverage = _check_label_contract(cfg, "ego4d")
    labels = coverage["labels"]
    assert labels["social_native.looking_at_wearer_subframes"]["materialized"]
    assert labels["social_native.talking_to_wearer_subframes"]["materialized"]
    assert not labels["text.speech_rate"]["supported"]
    assert coverage["sanity"]["floor_transfer_count"] > 0
