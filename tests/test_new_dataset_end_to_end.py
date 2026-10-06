"""A new dataset goes from raw annotations to a release without touching generic code.

``moodlab`` is a synthetic corpus in a raw format neither EgoCom nor Ego4D has
(one CSV of per-participant speech intervals, one CSV of sessions). The test
writes its adapters as any new dataset would, registers them, and runs the
whole canonical pipeline through the real CLI: cleaning, native and control
voice state, action grid, media manifest, model-ready index, label sidecars,
label audit and the Hugging Face release.

It is the contract behind "adding a dataset means writing its adapters".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from conv_wm import cli
from conv_wm.config import DEFAULT_CONFIG_PATH, get_path
from conv_wm.data import datasets
from conv_wm.data.datasets.spec import DatasetSpec, parquet_table
from conv_wm.data.labels import facts
from conv_wm.data.labels.facts import (
    LabelFactsSpec,
    LabelSource,
    ParticipantFacts,
    RecordingFacts,
)
from conv_wm.data.media.media_manifest import MediaFileIndex, MediaRecord, MediaSource
from conv_wm.data.pipeline.clean import CleanedAnnotationTable
from conv_wm.data.vocal.native_source import (
    NativeFocalVoiceSource,
    require_table,
    timed_intervals,
)
from conv_wm.data.vocal.native_state import NativeFocalRecording

DATASET = "moodlab"
RULE = "moodlab-cleaning-v1"
SCHEMA = "moodlab-speech-intervals-v1"
SESSIONS = {"s1": "train", "s2": "val", "s3": "test"}
DURATION_S = 120.0


# -- The raw corpus -----------------------------------------------------------


def _speech_rows() -> list[dict[str, object]]:
    """Wearer "a" and partner "b" alternate turns, with short gaps and overlaps."""
    rows = []
    for session in SESSIONS:
        t = 1.0
        for turn in range(40):
            speaker = "a" if turn % 2 == 0 else "b"
            length = 1.2 + 0.1 * (turn % 5)
            rows.append(
                {
                    "session": session,
                    "participant": speaker,
                    "start": t,
                    "end": t + length,
                }
            )
            t += length + (0.4 if turn % 3 else -0.2)  # gaps, and a few overlaps
            if t > DURATION_S - 3:
                break
    return rows


def _write_raw(raw: Path) -> None:
    annotations = raw / "MoodLab" / "annotations"
    media = raw / "MoodLab" / "media"
    annotations.mkdir(parents=True)
    media.mkdir(parents=True)
    pd.DataFrame(_speech_rows()).to_csv(annotations / "speech.csv", index=False)
    pd.DataFrame(
        {
            "session": list(SESSIONS),
            "wearer": ["a"] * len(SESSIONS),
            "duration_s": [DURATION_S] * len(SESSIONS),
            "split": list(SESSIONS.values()),
        }
    ).to_csv(annotations / "sessions.csv", index=False)
    for session in SESSIONS:
        (media / f"{session}.wav").write_bytes(b"")


def _write_media_metadata(cfg: DictConfig) -> None:
    """The media audit's table, as `conv-wm audit media` would write it."""
    path = Path(cfg.paths.reports) / "temporal" / "media_metadata"
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": [DATASET] * len(SESSIONS),
            "relative_path": [f"MoodLab/media/{s}.wav" for s in SESSIONS],
            "probe_ok": [True] * len(SESSIONS),
            "n_audio_streams": [1] * len(SESSIONS),
            "audio_start_time_sec": [0.0] * len(SESSIONS),
            "audio_duration_sec": [DURATION_S] * len(SESSIONS),
        }
    ).to_parquet(path / "media_metadata.parquet", index=False)


# -- The adapters a new dataset writes ----------------------------------------


def _clean(cfg: DictConfig) -> list[CleanedAnnotationTable]:
    tables = []
    for name, columns in (
        ("speech", ("session", "participant", "start", "end")),
        ("sessions", ("session", "wearer", "duration_s", "split")),
    ):
        source = get_path(cfg, DATASET, "raw", name)
        table = pd.read_csv(
            source, dtype={"session": "string", "participant": "string"}
        )
        tables.append(
            CleanedAnnotationTable(
                dataset=DATASET,
                name=name,
                output_key=f"{name}_clean",
                source_rows=len(table),
                table=table,
                decision="NO FILTER NEEDED",
                reason="synthetic source, every row is usable",
                required_fields=columns,
                optional_nullable_fields=(),
                removed_by_reason={},
                cleaning_rule=RULE,
                source_paths=(source,),
            )
        )
    return tables


def _tables(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, Path]]:
    speech_path = get_path(cfg, DATASET, "interim", "speech_clean")
    sessions_path = get_path(cfg, DATASET, "interim", "sessions_clean")
    return (
        require_table(speech_path),
        require_table(sessions_path),
        (
            speech_path,
            sessions_path,
        ),
    )


def _native_voice(cfg: DictConfig, media: pd.DataFrame) -> NativeFocalVoiceSource:
    del media  # every session's audio covers the whole session here
    speech, sessions, paths = _tables(cfg)
    recordings = []
    for row in sessions.to_dict(orient="records"):
        own = speech.loc[
            speech["session"].eq(row["session"])
            & speech["participant"].eq(row["wearer"])
        ]
        recordings.append(
            NativeFocalRecording(
                dataset=DATASET,
                recording_id=str(row["session"]),
                sync_group_id=str(row["session"]),
                view_id=str(row["session"]),
                wearer_id=str(row["wearer"]),
                start_s=0.0,
                end_s=float(row["duration_s"]),
                speaking=timed_intervals(own, "speech_clean", "start", "end"),
                unknown=(),
                source_kind="moodlab_speech_intervals",
                annotation_schema_version=SCHEMA,
            )
        )
    return NativeFocalVoiceSource(
        dataset=DATASET,
        recordings=recordings,
        annotation_paths=paths,
        annotation_schema_version=SCHEMA,
        cleaning_rule_version=RULE,
        statistics={"sessions": len(recordings)},
        limitations=("synthetic corpus",),
    )


def _media(
    cfg: DictConfig, media: pd.DataFrame, recording_ids: Sequence[str]
) -> MediaSource:
    del cfg
    files = MediaFileIndex(media, DATASET)
    records = []
    for recording_id in sorted(set(recording_ids)):
        video, audio = files.resolve(recording_id)
        records.append(MediaRecord(recording_id, video, audio, media_offset_s=0.0))
    return MediaSource(
        dataset=DATASET,
        records=tuple(records),
        unresolved={},
        mapping="session = file stem",
    )


def _splits(cfg: DictConfig) -> pd.DataFrame:
    sessions = parquet_table(DATASET, "interim", "sessions_clean")(cfg)
    return pd.DataFrame(
        {
            "recording_id": sessions["session"].astype("string"),
            "split": sessions["split"].astype("string"),
        }
    )


def _label_facts(cfg: DictConfig, media: pd.DataFrame) -> LabelSource:
    del media
    speech, sessions, paths = _tables(cfg)
    recordings = []
    for row in sessions.to_dict(orient="records"):
        session = str(row["session"])
        rows = speech.loc[speech["session"].eq(session)]
        recordings.append(
            RecordingFacts(
                dataset=DATASET,
                recording_id=session,
                conversation_id=session,
                view_id=session,
                wearer_id=str(row["wearer"]),
                start_s=0.0,
                end_s=float(row["duration_s"]),
                participants=tuple(
                    ParticipantFacts(
                        participant_id=pid,
                        is_wearer=pid == row["wearer"],
                        speech=timed_intervals(
                            rows.loc[rows["participant"].eq(pid)],
                            "speech_clean",
                            "start",
                            "end",
                        ),
                    )
                    for pid in sorted(rows["participant"].unique())
                ),
                unknown=(),
            )
        )
    return LabelSource(
        dataset=DATASET,
        recordings=recordings,
        annotation_paths=paths,
        annotation_schema_version=SCHEMA,
        cleaning_rule_version=RULE,
        tokens=pd.DataFrame(columns=list(facts.TOKEN_COLUMNS)),
        statistics={"recordings": len(recordings)},
        limitations=("synthetic corpus",),
    )


MOODLAB = DatasetSpec(
    name=DATASET,
    description="Synthetic two-person sessions (test only).",
    annotation_cleaner=_clean,
    native_focal_voice=_native_voice,
    media_records=_media,
    recording_splits=_splits,
    label_facts=LabelFactsSpec(provides=frozenset({facts.SPEECH}), load=_label_facts),
)


# -- The run ----------------------------------------------------------------


@pytest.fixture
def moodlab(tmp_path, monkeypatch):
    monkeypatch.setenv("EGO_DATA_ROOT", str(tmp_path))
    cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    cfg.datasets[DATASET] = {
        "raw": "${paths.raw}/MoodLab/annotations",
        "interim": "${paths.interim}/moodlab/annotations",
        "processed": "${paths.processed}/moodlab",
        "model_ready": "${paths.model_ready}/moodlab",
        "reports": "${paths.reports}/moodlab",
        "files": {
            "speech": "speech.csv",
            "sessions": "sessions.csv",
            "speech_clean": "speech_clean.parquet",
            "sessions_clean": "sessions_clean.parquet",
        },
    }
    cfg.release.public = [DATASET]
    cfg.release.full = [DATASET]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))
    resolved = OmegaConf.load(config_path)
    assert isinstance(resolved, DictConfig)
    _write_raw(Path(resolved.paths.raw))
    _write_media_metadata(resolved)
    datasets.register(MOODLAB)
    try:
        yield resolved, config_path
    finally:
        datasets.unregister(DATASET)


def _run(config_path: Path, *args: str) -> None:
    assert cli.main(["--config", str(config_path), *args]) == 0, args


def test_a_new_dataset_runs_from_raw_annotations_to_the_release(moodlab):
    cfg, config_path = moodlab

    _run(config_path, "audit", "manifest")
    _run(config_path, "clean", "annotations", "--dataset", DATASET)
    _run(config_path, "build", "all", "--dataset", DATASET)
    _run(config_path, "audit", "labels", "--dataset", DATASET)
    _run(config_path, "release")

    grid = pd.read_parquet(
        Path(cfg.paths.processed)
        / "vocal_action_grid"
        / DATASET
        / "vocal_action_grid.parquet"
    )
    assert set(grid["recording_id"]) == set(SESSIONS)
    assert {"ONSET", "OFFSET"} <= set(grid["action"].dropna())

    index = pd.read_parquet(Path(cfg.paths.model_ready) / DATASET / "index.parquet")
    by_split = index.groupby("split")["recording_id"].unique().map(set).to_dict()
    # The dataset's own splits are propagated, one session each.
    assert by_split == {"train": {"s1"}, "validation": {"s2"}, "test": {"s3"}}

    speech = pd.read_parquet(
        Path(cfg.paths.processed) / "labels" / DATASET / "speech" / "grid.parquet"
    )
    assert set(speech["recording_id"]) == set(SESSIONS)
    assert speech["others_active"].any()

    release = Path(cfg.release.output) / DATASET
    metadata = json.loads((release / "metadata.json").read_text())
    assert metadata["dataset"] == DATASET
    assert (release / "data" / "action_grid.parquet").is_file()
    for split in ("train", "validation", "test"):
        assert (release / "data" / "model_ready" / f"{split}.parquet").is_file()


def _with_grid_step(cfg: DictConfig, config_path: Path, step_s: float) -> None:
    cfg.grid.decision_step_s = step_s
    config_path.write_text(OmegaConf.to_yaml(cfg))


def test_the_decision_grid_step_is_configurable_end_to_end(moodlab, capsys):
    cfg, config_path = moodlab
    _with_grid_step(cfg, config_path, 0.08)  # 12.5 Hz, Mimi's frame rate

    _run(config_path, "audit", "manifest")
    _run(config_path, "clean", "annotations", "--dataset", DATASET)
    _run(config_path, "build", "all", "--dataset", DATASET)
    _run(config_path, "audit", "labels", "--dataset", DATASET)
    _run(config_path, "release")

    processed = Path(cfg.paths.processed)
    grid = pd.read_parquet(
        processed / "vocal_action_grid" / DATASET / "vocal_action_grid.parquet"
    )
    assert grid["decision_time_s"].to_numpy() == pytest.approx(
        grid["decision_index"].to_numpy() * 0.08
    )
    # Sessions last 120 s: 1,500 cells of 80 ms each.
    assert grid.groupby("recording_id").size().eq(1500).all()

    metadata = json.loads(
        (Path(cfg.paths.model_ready) / DATASET / "metadata.json").read_text()
    )
    assert metadata["grid"]["frequency_hz"] == 12.5
    assert metadata["windows"]["future_steps"] == 13  # 1 s, rounded up to 1.04 s

    speech = json.loads(
        (processed / "labels" / DATASET / "speech" / "manifest.json").read_text()
    )
    assert speech["time"]["decision_step_s"] == 0.08

    # A layer built at 80 ms is refused by a configuration asking for 100 ms.
    _with_grid_step(cfg, config_path, 0.1)
    capsys.readouterr()
    refused = cli.main(
        ["--config", str(config_path), "build", "model-ready", "--dataset", DATASET]
    )
    assert refused != 0
    assert "built on a 0.08 s decision grid" in capsys.readouterr().err


def test_a_registered_dataset_is_offered_by_the_cli(moodlab):
    _, config_path = moodlab

    help_text = cli.build_parser().format_help()
    assert cli.main(["--config", str(config_path), "datasets"]) == 0
    assert "build" in help_text
    with pytest.raises(SystemExit):  # an unknown dataset is still refused
        cli.main(
            ["--config", str(config_path), "build", "model-ready", "--dataset", "nope"]
        )
