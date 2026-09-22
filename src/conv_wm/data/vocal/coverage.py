"""Coverage of detected focal voice activity by the focal annotation.

Coverage and alignment are separate measurements: overlap ratios use the focal
annotation dilated by a versioned tolerance, so a slightly shifted annotation
still covers a segment, while the boundary offsets of covered segments are
reported on their own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from conv_wm.data.vocal.config import CoverageConfig
from conv_wm.data.vocal.identity import SegmentContext
from conv_wm.data.vocal.intervals import (
    Intervals,
    dilate,
    merge,
    nearest_boundary_offsets,
    overlap_duration,
    overlap_ratio,
    subtract,
    total_duration,
)
from conv_wm.data.vocal.records import (
    CoverageStatus,
    DetectedSegment,
    FocalRecording,
    FocalVoiceActivityStatus,
    IdentityDecision,
    RecordingCoverageSummary,
    SegmentAssessment,
    SpeakerAttribution,
    intervals_from_segments,
)

Attributor = Callable[[SegmentContext], IdentityDecision]


@dataclass(frozen=True)
class RecordingAssessment:
    """Per-segment assessments and the recording-level summary."""

    segments: list[SegmentAssessment]
    summary: RecordingCoverageSummary


def coverage_status(
    attribution: SpeakerAttribution, focal_ratio: float, config: CoverageConfig
) -> CoverageStatus:
    """Status of a focal-attributed segment from its dilated overlap ratio."""
    if attribution == SpeakerAttribution.OTHER:
        return CoverageStatus.NOT_FOCAL
    if attribution == SpeakerAttribution.UNRESOLVED:
        return CoverageStatus.UNRESOLVED_IDENTITY
    if focal_ratio >= config.covered_min_overlap_ratio:
        return CoverageStatus.COVERED
    if focal_ratio <= config.uncovered_max_overlap_ratio:
        return CoverageStatus.UNCOVERED
    return CoverageStatus.PARTIALLY_COVERED


def assess_recording(
    recording: FocalRecording,
    segments: list[DetectedSegment],
    *,
    attribute: Attributor,
    config: CoverageConfig,
    detection_method: str,
    identity_method: str,
    identity_confidence: float,
    focal_activity_measurable: bool = True,
    identity_diagnostics: dict[str, float] | None = None,
    decoded_duration_error_s: float = math.nan,
) -> RecordingAssessment:
    """Attribute and assess every detected segment, then summarize the recording."""
    focal = merge(recording.focal_annotation, gap_s=config.annotation_merge_gap_s)
    others = merge(recording.other_annotation, gap_s=config.annotation_merge_gap_s)
    tolerance = config.annotation_overlap_tolerance_s
    focal_dilated = dilate(focal, tolerance)
    others_dilated = dilate(others, tolerance)
    anyone_dilated = merge(np.concatenate([focal_dilated, others_dilated]))

    assessments: list[SegmentAssessment] = []
    for segment in sorted(segments, key=lambda s: (s.start_s, s.end_s)):
        start, end = segment.start_s, segment.end_s
        focal_ratio = overlap_ratio(start, end, focal_dilated)
        exact_ratio = overlap_ratio(start, end, focal)
        other_ratio = overlap_ratio(start, end, others_dilated)
        if recording.annotation_available:
            decision = attribute(SegmentContext(segment, focal_ratio, other_ratio))
            status = coverage_status(decision.attribution, focal_ratio, config)
        else:
            decision = IdentityDecision(
                SpeakerAttribution.UNRESOLVED, "none", math.nan, {}
            )
            status = CoverageStatus.ANNOTATION_UNAVAILABLE
        onset, offset = (
            nearest_boundary_offsets(start, end, focal)
            if status == CoverageStatus.COVERED
            else (math.nan, math.nan)
        )
        assessments.append(
            SegmentAssessment(
                dataset=recording.dataset,
                recording_id=recording.recording_id,
                sync_group_id=recording.sync_group_id,
                view_id=recording.view_id,
                wearer_id=recording.wearer_id,
                canonical_start_s=start,
                canonical_end_s=end,
                duration_s=segment.duration_s,
                annotation_overlap_ratio=focal_ratio,
                annotation_overlap_ratio_exact=exact_ratio,
                other_annotation_overlap_ratio=other_ratio,
                detection_method=detection_method,
                detection_confidence=segment.detection_confidence,
                identity_method=decision.method,
                identity_confidence=decision.confidence,
                speaker_attribution=decision.attribution,
                coverage_status=status,
                duration_bucket=config.bucket_of(segment.duration_s),
                onset_offset_s=onset,
                offset_offset_s=offset,
                uncovered_duration_s=segment.duration_s
                - overlap_duration(start, end, focal_dilated),
                identity_evidence=dict(decision.evidence),
            )
        )
    summary = summarize_recording(
        recording,
        assessments,
        focal=focal,
        focal_dilated=focal_dilated,
        anyone_dilated=anyone_dilated,
        config=config,
        detection_method=detection_method,
        identity_method=identity_method,
        identity_confidence=identity_confidence,
        focal_activity_measurable=focal_activity_measurable,
        identity_diagnostics=identity_diagnostics or {},
        decoded_duration_error_s=decoded_duration_error_s,
    )
    return RecordingAssessment(assessments, summary)


def summarize_recording(
    recording: FocalRecording,
    assessments: list[SegmentAssessment],
    *,
    focal: Intervals,
    focal_dilated: Intervals,
    anyone_dilated: Intervals,
    config: CoverageConfig,
    detection_method: str,
    identity_method: str,
    identity_confidence: float,
    focal_activity_measurable: bool,
    identity_diagnostics: dict[str, float],
    decoded_duration_error_s: float,
) -> RecordingCoverageSummary:
    """Aggregate segment assessments into the recording row."""
    detected = intervals_from_segments(
        [
            DetectedSegment(
                a.canonical_start_s, a.canonical_end_s, a.detection_confidence
            )
            for a in assessments
        ]
    )
    focal_segments = [
        a
        for a in assessments
        if a.speaker_attribution == SpeakerAttribution.FOCAL
        and a.coverage_status != CoverageStatus.ANNOTATION_UNAVAILABLE
    ]
    uncovered = [
        a for a in focal_segments if a.coverage_status == CoverageStatus.UNCOVERED
    ]
    partial = [
        a
        for a in focal_segments
        if a.coverage_status == CoverageStatus.PARTIALLY_COVERED
    ]
    unresolved = [
        a
        for a in assessments
        if a.coverage_status
        in (CoverageStatus.UNRESOLVED_IDENTITY, CoverageStatus.ANNOTATION_UNAVAILABLE)
    ]
    other = [a for a in assessments if a.coverage_status == CoverageStatus.NOT_FOCAL]
    covered_segments = [
        a for a in focal_segments if a.coverage_status == CoverageStatus.COVERED
    ]

    detected_duration = float(sum(a.duration_s for a in assessments))
    resolved_focal_duration = float(sum(a.duration_s for a in focal_segments))
    resolved_covered_duration = float(
        sum(a.duration_s - a.uncovered_duration_s for a in focal_segments)
    )
    resolved_uncovered_duration = float(
        sum(a.uncovered_duration_s for a in focal_segments)
    )
    focal_duration = resolved_focal_duration if focal_activity_measurable else math.nan
    covered_duration = (
        resolved_covered_duration if focal_activity_measurable else math.nan
    )
    uncovered_duration = (
        resolved_uncovered_duration if focal_activity_measurable else math.nan
    )
    labels = config.bucket_labels()
    counts = {label: 0 for label in labels}
    durations = {label: 0.0 for label in labels}
    for a in uncovered:
        counts[a.duration_bucket] += 1
        durations[a.duration_bucket] += a.duration_s
    short = sum(
        1 for a in uncovered if a.duration_s < config.short_segment_max_duration_s
    )
    onsets = np.array([a.onset_offset_s for a in covered_segments], dtype=float)
    offsets = np.array([a.offset_offset_s for a in covered_segments], dtype=float)
    return RecordingCoverageSummary(
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        sync_group_id=recording.sync_group_id,
        view_id=recording.view_id,
        wearer_id=recording.wearer_id,
        canonical_start_s=recording.canonical_start_s,
        canonical_end_s=recording.canonical_end_s,
        audited_duration_s=recording.duration_s,
        annotation_available=recording.annotation_available,
        annotation_source=recording.annotation_source,
        detection_method=detection_method,
        identity_method=identity_method,
        identity_confidence=identity_confidence,
        focal_voice_activity_status=(
            FocalVoiceActivityStatus.MEASURABLE
            if focal_activity_measurable
            else FocalVoiceActivityStatus.UNRESOLVED
        ),
        detected_acoustic_voice_duration_s=detected_duration,
        detected_segment_count=len(assessments),
        annotated_focal_voice_duration_s=total_duration(focal),
        detected_focal_voice_duration_s=focal_duration,
        focal_annotation_coverage_ratio=(
            covered_duration / focal_duration
            if focal_activity_measurable and focal_duration > 0
            else math.nan
        ),
        covered_focal_duration_s=covered_duration,
        uncovered_focal_voice_duration_s=uncovered_duration,
        uncovered_focal_segment_count=len(uncovered),
        partially_covered_focal_segment_count=len(partial),
        uncovered_short_segment_ratio=(
            short / len(uncovered) if uncovered else math.nan
        ),
        uncovered_segment_count_by_bucket=counts,
        uncovered_duration_s_by_bucket=durations,
        other_attributed_duration_s=float(sum(a.duration_s for a in other)),
        unresolved_identity_duration_s=float(sum(a.duration_s for a in unresolved)),
        unresolved_identity_segment_count=len(unresolved),
        unresolved_identity_ratio=(
            float(sum(a.duration_s for a in unresolved)) / detected_duration
            if detected_duration > 0
            else math.nan
        ),
        unannotated_any_speaker_duration_s=(
            total_duration(subtract(detected, anyone_dilated))
            if recording.annotation_available
            else math.nan
        ),
        annotated_focal_without_detection_s=(
            total_duration(
                subtract(focal, dilate(detected, config.annotation_overlap_tolerance_s))
            )
            if recording.annotation_available
            else math.nan
        ),
        onset_offset_s_median=float(np.median(onsets)) if len(onsets) else math.nan,
        offset_offset_s_median=float(np.median(offsets)) if len(offsets) else math.nan,
        onset_offset_s_abs_p95=float(np.quantile(np.abs(onsets), 0.95))
        if len(onsets)
        else math.nan,
        identity_precision_focal=identity_diagnostics.get("precision_focal", math.nan),
        identity_precision_other=identity_diagnostics.get("precision_other", math.nan),
        identity_recall_focal=identity_diagnostics.get("recall_focal", math.nan),
        identity_undecided_ratio=identity_diagnostics.get("undecided_ratio", math.nan),
        identity_validation_focal_count=int(
            identity_diagnostics.get("n_validation_focal", 0.0)
        ),
        identity_validation_other_count=int(
            identity_diagnostics.get("n_validation_other", 0.0)
        ),
        identity_synchronized_other_view_count=int(
            identity_diagnostics.get("n_synchronized_other_views", 0.0)
        ),
        identity_excluded_other_view_count=int(
            identity_diagnostics.get("n_excluded_other_views", 0.0)
        ),
        decoded_duration_error_s=decoded_duration_error_s,
    )
