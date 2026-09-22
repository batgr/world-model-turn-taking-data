import math

import numpy as np
import pytest
from helpers import assess, make_recording, seg

from conv_wm.data.vocal.records import (
    CoverageStatus,
    FocalVoiceActivityStatus,
    SpeakerAttribution,
)


def _only(result, status):
    return [a for a in result.segments if a.coverage_status == status]


def test_annotated_focal_vocalisation_is_covered(annotation_pipeline, config):
    recording = make_recording(focal=[(10.0, 12.0)])

    result = assess(recording, [seg(10.05, 11.95)], annotation_pipeline, config)

    [assessment] = result.segments
    assert assessment.speaker_attribution == SpeakerAttribution.FOCAL
    assert assessment.coverage_status == CoverageStatus.COVERED
    assert assessment.identity_method == "focal_annotation"
    assert result.summary.focal_annotation_coverage_ratio == pytest.approx(1.0)
    assert result.summary.uncovered_focal_segment_count == 0


def test_focal_vocalisation_without_annotation_is_uncovered_when_identity_is_known(
    config,
):
    """A dataset whose identity method resolves the speaker reports missing annotation."""
    from conv_wm.data.vocal.identity import IdentityPipeline
    from conv_wm.data.vocal.records import IdentityDecision

    recording = make_recording(focal=[(10.0, 12.0)])
    pipeline = IdentityPipeline(("focal_annotation",), config.identity)

    def attribute(segment):
        decision = pipeline.attribute(
            __import__(
                "conv_wm.data.vocal.identity", fromlist=["RecordingIdentityContext"]
            ).RecordingIdentityContext(pipeline.method_names),
            segment,
        )
        if decision.attribution == SpeakerAttribution.UNRESOLVED:
            return IdentityDecision(SpeakerAttribution.FOCAL, "oracle", 1.0, {})
        return decision

    from conv_wm.data.vocal.coverage import assess_recording

    result = assess_recording(
        recording,
        [seg(10.0, 12.0), seg(30.0, 30.4)],
        attribute=attribute,
        config=config.coverage,
        detection_method="static",
        identity_method="oracle",
        identity_confidence=1.0,
    )

    uncovered = _only(result, CoverageStatus.UNCOVERED)
    assert len(uncovered) == 1
    assert uncovered[0].canonical_start_s == 30.0
    assert uncovered[0].identity_method == "oracle"
    assert result.summary.uncovered_focal_segment_count == 1
    assert result.summary.uncovered_focal_voice_duration_s == pytest.approx(0.4)
    assert result.summary.focal_annotation_coverage_ratio == pytest.approx(2.0 / 2.4)


def test_slightly_shifted_annotation_stays_covered(annotation_pipeline, config):
    """Annotation shifted by less than the tolerance is alignment, not a missing annotation."""
    tolerance = config.coverage.annotation_overlap_tolerance_s
    recording = make_recording(
        focal=[(10.0 + tolerance * 0.75, 12.0 + tolerance * 0.75)]
    )

    result = assess(recording, [seg(10.0, 12.0)], annotation_pipeline, config)

    [assessment] = result.segments
    assert assessment.coverage_status == CoverageStatus.COVERED
    assert assessment.annotation_overlap_ratio == pytest.approx(1.0)
    assert assessment.annotation_overlap_ratio_exact < 1.0
    assert assessment.onset_offset_s == pytest.approx(-tolerance * 0.75)
    assert result.summary.onset_offset_s_median == pytest.approx(-tolerance * 0.75)


def test_other_participant_speech_is_not_a_missing_focal_annotation(
    annotation_pipeline, config
):
    recording = make_recording(focal=[(10.0, 12.0)], other=[(20.0, 23.0)])

    result = assess(recording, [seg(20.1, 22.9)], annotation_pipeline, config)

    [assessment] = result.segments
    assert assessment.speaker_attribution == SpeakerAttribution.OTHER
    assert assessment.coverage_status == CoverageStatus.NOT_FOCAL
    assert result.summary.uncovered_focal_segment_count == 0
    assert result.summary.uncovered_focal_voice_duration_s == 0.0
    assert result.summary.other_attributed_duration_s == pytest.approx(2.8)


def test_unattributable_speech_is_unresolved_identity(annotation_pipeline, config):
    recording = make_recording(focal=[(10.0, 12.0)], other=[(20.0, 23.0)])

    result = assess(recording, [seg(40.0, 41.0)], annotation_pipeline, config)

    [assessment] = result.segments
    assert assessment.speaker_attribution == SpeakerAttribution.UNRESOLVED
    assert assessment.coverage_status == CoverageStatus.UNRESOLVED_IDENTITY
    assert math.isnan(assessment.identity_confidence)
    assert result.summary.unresolved_identity_duration_s == pytest.approx(1.0)
    assert result.summary.unresolved_identity_segment_count == 1
    assert result.summary.uncovered_focal_segment_count == 0
    assert result.summary.unannotated_any_speaker_duration_s == pytest.approx(1.0)


def test_very_short_uncovered_segment_lands_in_the_shortest_bucket(config):
    from conv_wm.data.vocal.coverage import assess_recording
    from conv_wm.data.vocal.records import IdentityDecision

    recording = make_recording(focal=[(10.0, 12.0)])
    result = assess_recording(
        recording,
        [seg(30.0, 30.2), seg(35.0, 35.4), seg(40.0, 40.7), seg(45.0, 47.0)],
        attribute=lambda s: IdentityDecision(
            SpeakerAttribution.FOCAL, "oracle", 1.0, {}
        ),
        config=config.coverage,
        detection_method="static",
        identity_method="oracle",
        identity_confidence=1.0,
    )

    assert [a.duration_bucket for a in result.segments] == [
        "<0.25s",
        "0.25-0.5s",
        "0.5-1.0s",
        ">=1.0s",
    ]
    assert result.summary.uncovered_segment_count_by_bucket == {
        "<0.25s": 1,
        "0.25-0.5s": 1,
        "0.5-1.0s": 1,
        ">=1.0s": 1,
    }
    assert result.summary.uncovered_short_segment_ratio == pytest.approx(0.5)
    row = result.summary.to_row()
    assert row["uncovered_count_lt_0p25s"] == 1
    assert row["uncovered_duration_s_ge_1p0s"] == pytest.approx(2.0)


def test_missing_annotation_coverage_is_never_silence(annotation_pipeline, config):
    recording = make_recording(annotation_available=False)

    result = assess(
        recording, [seg(5.0, 6.0), seg(20.0, 20.3)], annotation_pipeline, config
    )

    assert {a.coverage_status for a in result.segments} == {
        CoverageStatus.ANNOTATION_UNAVAILABLE
    }
    assert result.summary.annotation_available is False
    assert result.summary.uncovered_focal_segment_count == 0
    assert result.summary.unresolved_identity_segment_count == 2
    assert math.isnan(result.summary.focal_annotation_coverage_ratio)
    assert math.isnan(result.summary.unannotated_any_speaker_duration_s)
    assert result.summary.detected_acoustic_voice_duration_s == pytest.approx(1.3)


def test_focal_segment_partially_covered_keeps_its_uncovered_remainder(
    config,
):
    from conv_wm.data.vocal.coverage import assess_recording
    from conv_wm.data.vocal.records import IdentityDecision

    recording = make_recording(focal=[(10.0, 11.0)])
    result = assess_recording(
        recording,
        [seg(10.0, 13.0)],
        attribute=lambda _: IdentityDecision(
            SpeakerAttribution.FOCAL, "oracle", 1.0, {}
        ),
        config=config.coverage,
        detection_method="static",
        identity_method="oracle",
        identity_confidence=1.0,
    )

    [assessment] = result.segments
    assert assessment.coverage_status == CoverageStatus.PARTIALLY_COVERED
    assert assessment.uncovered_duration_s == pytest.approx(3.0 - 1.2)
    assert result.summary.partially_covered_focal_segment_count == 1
    assert result.summary.uncovered_focal_voice_duration_s == pytest.approx(1.8)


def test_annotated_speech_without_detection_is_reported(annotation_pipeline, config):
    recording = make_recording(focal=[(10.0, 12.0), (30.0, 31.0)])

    result = assess(recording, [seg(10.0, 12.0)], annotation_pipeline, config)

    assert result.summary.annotated_focal_without_detection_s == pytest.approx(1.0)


def test_word_level_annotation_is_merged_into_utterances(annotation_pipeline, config):
    gap = config.coverage.annotation_merge_gap_s
    words = [(10.0, 10.3), (10.3 + gap * 0.5, 10.9), (10.9 + gap * 0.5, 11.5)]
    recording = make_recording(focal=words)

    result = assess(recording, [seg(10.0, 11.5)], annotation_pipeline, config)

    assert result.summary.annotated_focal_voice_duration_s == pytest.approx(1.5)
    assert result.segments[0].coverage_status == CoverageStatus.COVERED


def test_deterministic_for_identical_inputs(annotation_pipeline, config):
    rng = np.random.default_rng(7)
    starts = np.sort(rng.uniform(0, 55, 40))
    segments = [seg(float(s), float(s) + float(rng.uniform(0.1, 2.0))) for s in starts]
    recording = make_recording(focal=[(5.0, 9.0), (20.0, 25.0)], other=[(30.0, 40.0)])

    first = assess(recording, segments, annotation_pipeline, config)
    second = assess(recording, list(reversed(segments)), annotation_pipeline, config)

    assert [a.to_row() for a in first.segments] == [a.to_row() for a in second.segments]
    assert first.summary.to_row() == second.summary.to_row()


def test_annotation_only_identity_does_not_claim_focal_completeness(
    annotation_pipeline, config
):
    from conv_wm.data.vocal.coverage import assess_recording
    from conv_wm.data.vocal.identity import RecordingIdentityContext

    recording = make_recording(focal=[(10.0, 12.0)])
    context = RecordingIdentityContext(annotation_pipeline.method_names)
    result = assess_recording(
        recording,
        [seg(10.0, 12.0), seg(30.0, 31.0)],
        attribute=lambda segment: annotation_pipeline.attribute(context, segment),
        config=config.coverage,
        detection_method="static",
        identity_method=annotation_pipeline.name,
        identity_confidence=context.confidence,
        focal_activity_measurable=context.can_attribute_unannotated_focal,
    )

    assert result.segments[0].coverage_status == CoverageStatus.COVERED
    assert result.segments[1].coverage_status == CoverageStatus.UNRESOLVED_IDENTITY
    assert (
        result.summary.focal_voice_activity_status
        == FocalVoiceActivityStatus.UNRESOLVED
    )
    assert math.isnan(result.summary.focal_annotation_coverage_ratio)
    assert math.isnan(result.summary.detected_focal_voice_duration_s)


def test_segment_overlapping_focal_and_other_annotations_is_unresolved(
    annotation_pipeline, config
):
    recording = make_recording(focal=[(10.0, 12.0)], other=[(10.0, 12.0)])

    result = assess(recording, [seg(10.0, 12.0)], annotation_pipeline, config)

    [assessment] = result.segments
    assert assessment.coverage_status == CoverageStatus.UNRESOLVED_IDENTITY
    assert assessment.identity_method == "annotation_overlap_conflict"
