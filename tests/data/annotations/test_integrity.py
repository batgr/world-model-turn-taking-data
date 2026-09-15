import pandas as pd
import pytest

from conv_wm.data.annotations import (
    AnnotationProvenance,
    AnnotationScope,
    AnnotationSourceSpec,
    AnnotationSpec,
    CrossSourceComparison,
    DurationBounds,
    EntityReference,
    MediaReference,
    ReferenceKind,
    Severity,
    TemporalFields,
    TemporalOrigin,
    TimeUnit,
    audit_source,
    compare_sources,
)


def _loader(frame: pd.DataFrame):
    return lambda cfg: frame


def _spec(**overrides) -> AnnotationSourceSpec:
    base: dict = {
        "dataset": "synthetic",
        "name": "events",
        "description": "test intervals",
        "scope": AnnotationScope.TEMPORAL_INTERVAL,
        "load": _loader(pd.DataFrame()),
        "dimensions": ("speech",),
        "provenance": AnnotationProvenance.HUMAN_OBSERVED,
        "temporal": TemporalFields(
            "start", "end", TimeUnit.SECONDS, TemporalOrigin.MEDIA_START
        ),
    }
    base.update(overrides)
    return AnnotationSourceSpec(**base)


def _anomaly(report, constraint):
    return next(a for a in report.anomalies if a.constraint == constraint)


def test_ordered_finite_intervals_are_clean():
    table = pd.DataFrame({"start": [0.0, 1.0], "end": [0.5, 2.0]})

    report = audit_source(_spec(), table, tables={}, media=None)

    assert report.valid
    assert report.downstream_suitability == "usable"
    assert report.temporal is not None
    assert report.temporal.duration_quantiles["max"] == 1.0


def test_start_after_end_is_an_error_and_negative_start_a_warning():
    table = pd.DataFrame({"start": [-0.5, 3.0], "end": [0.5, 2.0]})

    report = audit_source(_spec(), table, tables={}, media=None)

    assert _anomaly(report, "start_after_end").count == 1
    assert _anomaly(report, "start_after_end").severity == Severity.ERROR
    assert _anomaly(report, "negative_start").count == 1
    assert _anomaly(report, "negative_start").severity == Severity.WARNING
    assert not report.valid


def test_nullable_timestamps_are_information_not_anomalies():
    table = pd.DataFrame({"start": [0.0, None], "end": [1.0, None]})
    nullable = TemporalFields(
        "start",
        "end",
        TimeUnit.SECONDS,
        TemporalOrigin.INTERACTION_START,
        nullable=True,
    )

    report = audit_source(_spec(temporal=nullable), table, tables={}, media=None)

    missing = _anomaly(report, "missing_timestamps")
    assert missing.count == 2
    assert missing.severity == Severity.INFO
    assert report.valid


def test_bounds_against_a_referenced_table_report_overshoot():
    clips = pd.DataFrame(
        {"clip": ["a", "b"], "clip_start": [0.0, 0.0], "clip_end": [10.0, 5.0]}
    )
    table = pd.DataFrame(
        {"clip": ["a", "b", "zzz"], "start": [1.0, 4.0, 0.0], "end": [9.0, 5.5, 1.0]}
    )
    spec = _spec(
        bounds=DurationBounds(
            source="clips",
            key_columns=("clip",),
            target_key_columns=("clip",),
            start_column="clip_start",
            end_column="clip_end",
        ),
    )

    report = audit_source(spec, table, tables={"clips": clips}, media=None)

    assert report.temporal is not None
    assert report.temporal.n_out_of_bounds == 1
    assert report.temporal.n_unbounded == 1
    assert report.temporal.max_overshoot == pytest.approx(0.5)
    assert (
        _anomaly(report, "end_beyond_reference_duration").severity == Severity.WARNING
    )


def test_entity_reference_ignores_unknown_sentinels_and_counts_orphans():
    persons = pd.DataFrame({"clip": ["a", "a"], "person": ["0", "1"]})
    table = pd.DataFrame(
        {
            "clip": ["a", "a", "a"],
            "person": ["1", "-1", "7"],
            "start": [0, 0, 0],
            "end": [1, 1, 1],
        }
    )
    spec = _spec(
        entity_references=(
            EntityReference(
                ReferenceKind.PARTICIPANT,
                ("clip", "person"),
                "persons",
                ("clip", "person"),
                ("-1",),
            ),
        )
    )

    report = audit_source(spec, table, tables={"persons": persons}, media=None)

    reference = report.references[0]
    assert reference.n_unknown == 1
    assert reference.n_orphans == 1
    assert not report.valid


def test_media_reference_uses_media_table_and_measured_duration():
    media = pd.DataFrame(
        {
            "dataset": ["synthetic"],
            "relative_path": ["Synthetic/v1.mp4"],
            "audio_duration_sec": [10.0],
            "video_duration_sec": [9.5],
        }
    )
    table = pd.DataFrame(
        {"video": ["v1", "v2"], "start": [0.0, 0.0], "end": [9.8, 1.0]}
    )
    spec = _spec(
        media_reference=MediaReference(columns=("video",)),
        bounds=DurationBounds("media", ("video",), ("relative_path",)),
    )

    report = audit_source(spec, table, tables={}, media=media)

    assert report.references[0].kind == "media"
    assert report.references[0].n_orphans == 1
    assert report.temporal is not None
    assert report.temporal.n_out_of_bounds == 1  # 9.8 > min(10.0, 9.5)
    assert report.temporal.n_unbounded == 1


def test_duplicate_identity_is_an_error():
    table = pd.DataFrame({"id": [1, 1], "start": [0.0, 1.0], "end": [0.5, 2.0]})

    report = audit_source(_spec(identity=("id",)), table, tables={}, media=None)

    assert _anomaly(report, "duplicate_identity").count == 2
    assert not report.valid


def test_missing_declared_columns_block_the_source():
    table = pd.DataFrame({"start": [0.0]})

    report = audit_source(_spec(), table, tables={}, media=None)

    assert _anomaly(report, "missing_declared_columns").detail == "end"
    assert report.downstream_suitability == "blocked"


def test_cross_source_overlap_coverage_is_directional():
    left_spec = _spec(name="voice")
    right_spec = _spec(name="talk")
    voice = pd.DataFrame(
        {
            "clip": ["a", "a", "a"],
            "person": ["1", "1", "2"],
            "start": [0, 5, 0],
            "end": [1, 6, 1],
        }
    )
    talk = pd.DataFrame({"clip": ["a"], "person": ["1"], "start": [0.5], "end": [0.8]})
    comparison = CrossSourceComparison(
        "voice_vs_talk", "voice", "talk", ("clip", "person"), ("clip", "person")
    )

    result = compare_sources(comparison, left_spec, right_spec, voice, talk)

    assert result.left_covered_fraction == pytest.approx(1 / 3)
    assert result.right_covered_fraction == pytest.approx(1.0)
    assert result.n_shared_keys == 1
    assert result.n_left_only_keys == 1


def test_annotation_spec_rejects_unknown_reference_targets():
    with pytest.raises(ValueError, match="unknown"):
        AnnotationSpec(
            sources=(
                _spec(
                    entity_references=(
                        EntityReference(
                            ReferenceKind.CLIP, ("clip",), "nowhere", ("clip",)
                        ),
                    )
                ),
            )
        )
