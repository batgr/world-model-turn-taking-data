"""A synthetic third dataset proves that adding a dataset touches no generic code.

``moodlab`` contributes two annotation types that neither Ego4D nor EgoCom has:
a video-level mood rating with participant traits, and confidence-scored affect
intervals. It registers structural contracts, annotation semantics and an
audio boundary grid, then flows through the generic structural and annotation
audits and the generic audio interpretation unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest
from omegaconf import OmegaConf

from conv_wm.data import datasets
from conv_wm.data.annotations import (
    AnnotationProvenance,
    AnnotationScope,
    AnnotationSourceSpec,
    AnnotationSpec,
    EntityReference,
    MediaReference,
    ReferenceKind,
    TemporalFields,
    TemporalOrigin,
    TimeUnit,
)
from conv_wm.data.audits.annotations import run_annotation_audit
from conv_wm.data.audits.structural import run_structural_audit
from conv_wm.data.datasets.spec import (
    AudioInterpretation,
    DatasetSpec,
    RelationSpec,
    StructuralSpec,
    TableSpec,
    csv_table,
)
from conv_wm.data.media.audio import (
    AudioTimelineEvent,
    EventCategory,
    EventDirection,
    EventType,
    KnownBoundaryGrid,
    interpret_events,
)

SESSIONS_SCHEMA = pa.DataFrameSchema(
    {
        "session_id": pa.Column(str, unique=True),
        "video_name": pa.Column(str),
        "participant_id": pa.Column(str),
        "mood_rating": pa.Column(int, checks=pa.Check.in_range(1, 7)),
        "age_group": pa.Column(str),
    },
    strict=True,
)
AFFECT_SCHEMA = pa.DataFrameSchema(
    {
        "session_id": pa.Column(str),
        "start_ms": pa.Column(float),
        "end_ms": pa.Column(float),
        "affect": pa.Column(str),
        "confidence": pa.Column(float, checks=pa.Check.in_range(0, 1)),
    },
    strict=True,
)


def moodlab_spec() -> DatasetSpec:
    return DatasetSpec(
        name="moodlab",
        description="Synthetic affect dataset used to prove extensibility.",
        structure=StructuralSpec(
            tables=(
                TableSpec(
                    "sessions", SESSIONS_SCHEMA, csv_table("moodlab", "raw", "sessions")
                ),
                TableSpec(
                    "affect", AFFECT_SCHEMA, csv_table("moodlab", "raw", "affect")
                ),
            ),
            relations=(
                RelationSpec(
                    "affect_session_in_sessions",
                    "affect",
                    ("session_id",),
                    "sessions",
                    ("session_id",),
                ),
            ),
        ),
        annotations=AnnotationSpec(
            sources=(
                AnnotationSourceSpec(
                    dataset="moodlab",
                    name="sessions",
                    description="One recording session: a self-reported mood and participant traits.",
                    scope=AnnotationScope.VIDEO,
                    load=csv_table("moodlab", "raw", "sessions"),
                    dimensions=("mood", "participant_traits"),
                    provenance=AnnotationProvenance.HUMAN_OBSERVED,
                    identity=("session_id",),
                    media_reference=MediaReference(columns=("video_name",)),
                    value_fields=("mood_rating", "age_group", "participant_id"),
                ),
                AnnotationSourceSpec(
                    dataset="moodlab",
                    name="affect",
                    description="Model-inferred affect intervals with a confidence score.",
                    scope=AnnotationScope.TEMPORAL_INTERVAL,
                    load=csv_table("moodlab", "raw", "affect"),
                    dimensions=("affect",),
                    provenance=AnnotationProvenance.MODEL_INFERRED,
                    temporal=TemporalFields(
                        "start_ms",
                        "end_ms",
                        TimeUnit.MILLISECONDS,
                        TemporalOrigin.MEDIA_START,
                    ),
                    entity_references=(
                        EntityReference(
                            ReferenceKind.INTERACTION,
                            ("session_id",),
                            "sessions",
                            ("session_id",),
                        ),
                    ),
                    value_fields=("affect",),
                    confidence_field="confidence",
                ),
            ),
        ),
        audio=AudioInterpretation(
            known_boundary_grid=KnownBoundaryGrid(period_sec=120.0)
        ),
    )


@pytest.fixture
def moodlab(tmp_path: Path):
    spec = datasets.register(moodlab_spec())
    raw = tmp_path / "raw" / "MoodLab"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "video_name": ["s1_cam", "s2_cam"],
            "participant_id": ["p1", "p2"],
            "mood_rating": [3, 6],
            "age_group": ["25-34", "35-44"],
        }
    ).to_csv(raw / "sessions.csv", index=False)
    pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "ghost"],
            "start_ms": [0.0, 500.0, 100.0, 0.0],
            "end_ms": [400.0, 900.0, 300.0, 10.0],
            "affect": ["neutral", "joy", "neutral", "joy"],
            "confidence": [0.9, 0.6, 0.8, 0.5],
        }
    ).to_csv(raw / "affect.csv", index=False)
    reports = tmp_path / "reports"
    media = pd.DataFrame(
        {
            "dataset": ["moodlab", "moodlab"],
            "relative_path": ["MoodLab/s1_cam.mp4", "MoodLab/s2_cam.mp4"],
            "audio_duration_sec": [1.0, 1.0],
            "video_duration_sec": [1.0, 1.0],
        }
    )
    (reports / "temporal" / "media_metadata").mkdir(parents=True)
    media.to_parquet(
        reports / "temporal" / "media_metadata" / "media_metadata.parquet", index=False
    )
    cfg = OmegaConf.create(
        {
            "paths": {
                stage: str(tmp_path / stage)
                for stage in ("raw", "interim", "validated", "processed", "model_ready")
            }
            | {"reports": str(reports)},
            "datasets": {
                "moodlab": {
                    "raw": str(raw),
                    "files": {"sessions": "sessions.csv", "affect": "affect.csv"},
                }
            },
        }
    )
    try:
        yield spec, cfg
    finally:
        datasets.unregister("moodlab")


def test_structural_audit_runs_the_new_dataset_through_generic_code(moodlab):
    _, cfg = moodlab

    outputs = run_structural_audit(cfg, dataset_names=["moodlab"])

    result = outputs.report["datasets"]["moodlab"]
    assert set(result["tables"]) == {"sessions", "affect"}
    assert result["tables"]["sessions"]["valid"]
    # The ghost session is an orphan relation, detected without any moodlab-specific code.
    assert result["relations"]["affect_session_in_sessions"]["orphan_rows"] == 1
    assert not outputs.valid


def test_annotation_audit_preserves_the_new_dataset_semantics(moodlab):
    _, cfg = moodlab

    outputs = run_annotation_audit(cfg, dataset_names=["moodlab"])

    contract = outputs.summary["datasets"]["moodlab"]["sources"]
    sessions, affect = contract["sessions"], contract["affect"]
    assert sessions["scope"] == "video"
    assert sessions["dimensions"] == ["mood", "participant_traits"]
    assert sessions["value_fields"] == ["mood_rating", "age_group", "participant_id"]
    assert sessions["media_relation"]["columns"] == ["video_name"]
    assert sessions["valid"]

    assert affect["provenance"] == "model_inferred"
    assert affect["confidence_field"] == "confidence"
    assert affect["temporal_coordinates"]["unit"] == "milliseconds"
    assert affect["temporal"]["duration_quantiles"]["max"] == 400.0
    orphan = next(r for r in affect["references"] if r["kind"] == "interaction")
    assert orphan["n_orphans"] == 1
    assert affect["downstream_suitability"] == "blocked"
    assert (outputs.output_dir / "annotation_sources.parquet").exists()


def test_generic_audio_interpretation_uses_the_declared_boundary_grid(moodlab):
    spec, _ = moodlab
    nan = math.nan
    event = AudioTimelineEvent(
        packet_index=0,
        next_packet_index=1,
        event_type=EventType.PCM_STEP_EPISODE,
        event_category=EventCategory.PCM_TIMESTAMP_VARIATION,
        event_direction=EventDirection.OVERLAP,
        event_pts=0,
        event_time_sec=240.01,
        event_end_time_sec=240.01,
        n_steps=1,
        magnitude_samples=1023.0,
        packet_timeline_error_samples=nan,
        first_pcm_step_error_samples=-1023.0,
        max_abs_pcm_step_error_samples=1023.0,
        net_pcm_drift_samples=-1023.0,
        robust_nominal_packet_duration_samples=1024.0,
        long_packet_duration_threshold_samples=1536.0,
        n_abnormally_long_packets=0,
        max_packet_duration_samples=1024.0,
        n_candidate_steps=0,
        n_compensated_candidate_steps=0,
        dropout_duration_samples=nan,
        dropout_duration_ms=nan,
        dropout_min_residual_in_window_samples=nan,
        dropout_residual_at_window_end_samples=nan,
        dropout_window_truncated=False,
        reference_decoded_samples=1024,
    )

    [interpreted] = interpret_events(
        [event],
        sample_rate_hz=48000,
        reference_decoded_samples=1024,
        skip_samples=2048,
        boundary_grid=datasets.get(spec.name).audio.known_boundary_grid,
    )

    assert interpreted.nearest_known_boundary_time_sec == 240.0
    assert interpreted.event.event_category == EventCategory.STITCH_RELATED


def test_dataset_listing_includes_the_registered_dataset(moodlab):
    assert "moodlab" in datasets.names()
    assert (
        datasets.get("moodlab").annotations.source("affect").confidence_field
        == "confidence"
    )
