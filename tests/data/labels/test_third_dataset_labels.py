"""A synthetic third corpus gets every label its facts allow, with no core change.

``triad`` is a three-person panel corpus unlike EgoCom and Ego4D: letter
speaker ids, a non-zero clock origin, utterance-level transcripts and no
social annotation. It registers a ``DatasetSpec`` with a native focal voice
adapter and label facts, then flows through the unchanged native, control,
grid and label builds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from label_helpers import recording
from omegaconf import OmegaConf

from conv_wm.data import datasets
from conv_wm.data.datasets.spec import DatasetSpec
from conv_wm.data.labels import facts, store
from conv_wm.data.labels.catalog import REGISTRY
from conv_wm.data.labels.facts import LabelFactsSpec, LabelSource
from conv_wm.data.labels.registry import support_matrix
from conv_wm.data.labels.selection import LabelSelection
from conv_wm.data.pipeline.action_grid import run_vocal_action_grid_build
from conv_wm.data.pipeline.control_state import run_control_focal_voice_state_build
from conv_wm.data.pipeline.labels import run_label_build
from conv_wm.data.pipeline.native_state import run_native_focal_voice_state_build
from conv_wm.data.vocal.native_source import NativeFocalVoiceSource
from conv_wm.data.vocal.native_state import NativeFocalRecording, SourceKind

SRC = Path(__file__).resolve().parents[3] / "src" / "conv_wm"


def triad_facts():
    return [
        recording(
            {
                "A": [(11.0, 13.0), (17.0, 18.0)],
                "B": [(13.5, 15.0)],
                "C": [(14.5, 16.0)],
            },
            wearer="A",
            start=10.0,
            end=20.0,
            recording_id=f"panel-{i}",
            dataset="triad",
        )
        for i in range(2)
    ]


def load_facts(cfg, media) -> LabelSource:
    del cfg, media
    tokens = pd.DataFrame(
        [
            {
                "recording_id": f"panel-{i}",
                "participant_id": "B",
                "unit": "utterance",
                "text": "so what now?",
                "start_s": 13.5,
                "end_s": 15.0,
                "timing_valid": True,
                "source_row": i,
            }
            for i in range(2)
        ]
    )
    return LabelSource(
        dataset="triad",
        recordings=triad_facts(),
        annotation_paths=(),
        annotation_schema_version="triad-v1",
        cleaning_rule_version="triad-clean-v1",
        tokens=tokens,
    )


def load_native(cfg, media) -> NativeFocalVoiceSource:
    del cfg, media
    return NativeFocalVoiceSource(
        dataset="triad",
        recordings=[
            NativeFocalRecording(
                dataset="triad",
                recording_id=item.recording_id,
                sync_group_id=item.conversation_id,
                view_id=item.view_id,
                wearer_id=item.wearer_id,
                start_s=item.start_s,
                end_s=item.end_s,
                speaking=next(p for p in item.participants if p.is_wearer).speech,
                unknown=item.unknown,
                source_kind=SourceKind.EGOCOM_TRANSCRIPT,
                annotation_schema_version="triad-v1",
            )
            for item in triad_facts()
        ],
        annotation_paths=(),
        annotation_schema_version="triad-v1",
        cleaning_rule_version="triad-clean-v1",
    )


@pytest.fixture
def triad(tmp_path):
    spec = datasets.register(
        DatasetSpec(
            name="triad",
            description="Synthetic three-person panel corpus.",
            native_focal_voice=load_native,
            label_facts=LabelFactsSpec(
                provides=frozenset({facts.SPEECH, facts.TRANSCRIPT}), load=load_facts
            ),
        )
    )
    reports = tmp_path / "reports"
    media = reports / "temporal" / "media_metadata" / "media_metadata.parquet"
    media.parent.mkdir(parents=True)
    pd.DataFrame({"dataset": ["triad"], "relative_path": ["Triad/x.mp4"]}).to_parquet(
        media
    )
    manifest = reports / "manifest" / "raw_manifest.parquet"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame({"relative_path": ["x"]}).to_parquet(manifest)
    cfg = OmegaConf.create(
        {
            "paths": {
                stage: str(tmp_path / stage)
                for stage in ("raw", "interim", "validated", "processed", "model_ready")
            }
            | {"reports": str(reports)},
            "datasets": {
                "triad": {"raw": str(tmp_path / "raw" / "Triad"), "files": {}}
            },
            "manifest": {"output": str(manifest)},
        }
    )
    try:
        yield spec, cfg
    finally:
        datasets.unregister("triad")


def test_the_third_dataset_flows_through_the_unchanged_label_build(triad):
    _, cfg = triad
    run_native_focal_voice_state_build(cfg, dataset="triad")
    run_control_focal_voice_state_build(cfg, dataset="triad")
    run_vocal_action_grid_build(cfg, dataset="triad")
    outputs = run_label_build(cfg, dataset="triad")
    assert {str(o.extractor) for o in outputs} == {"speech", "social", "text"}
    root = Path(cfg.paths.processed) / "labels" / "triad"
    bundle = store.load_labels(
        root,
        LabelSelection(
            True,
            (
                "instantaneous.speaker_activity",
                "next_speaker.*",
                "text.interrogative_cue",
            ),
        ),
    )
    grid = bundle["grid"].to_pandas()
    assert grid["participant_ids"].iloc[0].tolist() == ["A", "B", "C"]
    first = grid.decision_index.min()
    assert first == 100  # the clock starts at 10 s, the grid keeps it
    at_13 = grid.loc[grid.decision_index == 130].iloc[0]
    assert at_13.next_speaker == 1 and at_13.next_speaker_valid
    units = bundle["segments"].to_pandas()
    assert units.interrogative_cue.tolist() == [True, True]
    social = store.read_manifest(root, "social")
    assert social is not None
    assert "social_native.looking_at_wearer_subframes" in social["unavailable_labels"]


def test_support_follows_the_declared_facts():
    rows = support_matrix(REGISTRY, {"triad": {facts.SPEECH, facts.TRANSCRIPT}})
    supported = {row.label for row in rows if row.supported}
    assert "text.interrogative_cue" in supported
    assert "text.speech_rate" not in supported  # needs word timings
    assert "metadata.background_conditions" not in supported


GENERIC_MODULES = (
    *sorted((SRC / "data" / "labels").glob("*.py")),
    SRC / "data" / "pipeline" / "labels.py",
)


@pytest.mark.parametrize("path", GENERIC_MODULES, ids=lambda p: p.name)
def test_generic_label_code_never_names_a_dataset(path):
    source = path.read_text(encoding="utf-8")
    code = re.sub(r'"""[\s\S]*?"""', "", source)
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    assert not re.search(r"['\"](egocom|ego4d)['\"]", code, flags=re.IGNORECASE), path
