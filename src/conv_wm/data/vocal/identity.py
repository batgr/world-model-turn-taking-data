"""Speaker attribution of detected segments.

Methods are applied in order until one decides; a segment no method can decide
is ``UNRESOLVED`` and is never counted as a missing focal annotation. Every
decision records its method, its confidence and the numbers it was based on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from conv_wm.data.vocal.audio import (
    DecodedAudio,
    energy_envelope_db,
    envelope_lag_frames,
    shift_envelope,
)
from conv_wm.data.vocal.config import IdentityConfig, MultiDeviceEnergyConfig
from conv_wm.data.vocal.intervals import Intervals, as_intervals
from conv_wm.data.vocal.records import (
    DetectedSegment,
    FocalRecording,
    IdentityDecision,
    SpeakerAttribution,
)


@dataclass(frozen=True)
class SegmentContext:
    """What the coverage step already knows about a segment."""

    segment: DetectedSegment
    focal_overlap_ratio: float
    """Overlap with the (dilated) focal annotation."""
    other_overlap_ratio: float
    """Overlap with the (dilated) annotation of other participants."""


@dataclass
class RecordingIdentityContext:
    """Per-recording state prepared once (envelopes, synchronization, validation)."""

    method_names: tuple[str, ...]
    confidence: float = math.nan
    """Recording-level confidence of the strongest measurable method (NaN if none)."""
    energy: MultiDeviceEnergyState | None = None
    can_attribute_unannotated_focal: bool = False
    """Whether a method independent of focal annotation passed its validation gate."""
    notes: dict[str, float] = field(default_factory=dict)


@dataclass
class MultiDeviceEnergyState:
    """Aligned energy envelopes of the focal device and the other devices."""

    frame_s: float
    canonical_offset_s: float
    focal_db: np.ndarray
    others_db: dict[str, np.ndarray]
    """Other views aligned to the focal timeline (NaN where alignment has no data)."""
    lags_frames: dict[str, int]
    correlations: dict[str, float]
    excluded_views: tuple[str, ...]
    precision_focal: float = math.nan
    """Validated precision of 'focal' attributions on annotated intervals."""
    precision_other: float = math.nan
    recall_focal: float = math.nan
    undecided_ratio: float = math.nan
    n_validation_focal: int = 0
    n_validation_other: int = 0

    def dominance_db(self, start_s: float, end_s: float) -> float:
        """Focal mean energy minus the loudest other device over ``[start_s, end_s)``.

        NaN when no synchronized other device has data on the interval.
        """
        local_start = start_s - self.canonical_offset_s
        local_end = end_s - self.canonical_offset_s
        first = max(0, int(local_start / self.frame_s))
        last = min(
            len(self.focal_db), max(first + 1, math.ceil(local_end / self.frame_s))
        )
        focal = self.focal_db[first:last]
        if len(focal) == 0:
            return math.nan
        others = []
        for values in self.others_db.values():
            window = values[first:last]
            window = window[np.isfinite(window)]
            if len(window):
                others.append(float(window.mean()))
        if not others:
            return math.nan
        return float(focal.mean()) - max(others)


def focal_annotation_method(
    context: SegmentContext, config: IdentityConfig
) -> IdentityDecision | None:
    """A segment overlapping the focal annotation is the focal participant's."""
    if context.focal_overlap_ratio >= config.focal_annotation_min_overlap_ratio:
        if context.other_overlap_ratio >= config.other_annotation_min_overlap_ratio:
            return IdentityDecision(
                attribution=SpeakerAttribution.UNRESOLVED,
                method="annotation_overlap_conflict",
                confidence=math.nan,
                evidence={
                    "focal_overlap_ratio": context.focal_overlap_ratio,
                    "other_overlap_ratio": context.other_overlap_ratio,
                },
            )
        return IdentityDecision(
            attribution=SpeakerAttribution.FOCAL,
            method="focal_annotation",
            confidence=1.0,
            evidence={"focal_overlap_ratio": context.focal_overlap_ratio},
        )
    return None


def other_annotation_method(
    context: SegmentContext, config: IdentityConfig
) -> IdentityDecision | None:
    """A segment inside another participant's annotation is that participant's."""
    if context.other_overlap_ratio >= config.other_annotation_min_overlap_ratio:
        return IdentityDecision(
            attribution=SpeakerAttribution.OTHER,
            method="other_annotation",
            confidence=context.other_overlap_ratio,
            evidence={"other_overlap_ratio": context.other_overlap_ratio},
        )
    return None


def multi_device_energy_method(
    context: SegmentContext,
    state: MultiDeviceEnergyState | None,
    config: MultiDeviceEnergyConfig,
) -> IdentityDecision | None:
    """Attribute by energy dominance of the focal device over synchronized others.

    Requires the recording-level validation to have reached
    ``min_identity_confidence`` for the chosen class; otherwise the decision is
    ``UNRESOLVED`` with the measured dominance kept as evidence.
    """
    if state is None:
        return None
    segment = context.segment
    dominance = state.dominance_db(segment.start_s, segment.end_s)
    evidence = {"dominance_db": dominance}
    if math.isnan(dominance):
        return IdentityDecision(
            SpeakerAttribution.UNRESOLVED, "multi_device_energy", math.nan, evidence
        )
    if dominance >= config.dominance_threshold_db:
        attribution, confidence = SpeakerAttribution.FOCAL, state.precision_focal
    elif dominance <= -config.dominance_threshold_db:
        attribution, confidence = SpeakerAttribution.OTHER, state.precision_other
    else:
        return IdentityDecision(
            SpeakerAttribution.UNRESOLVED, "multi_device_energy", math.nan, evidence
        )
    if math.isnan(confidence) or confidence < config.min_identity_confidence:
        evidence["insufficient_validated_confidence"] = confidence
        return IdentityDecision(
            SpeakerAttribution.UNRESOLVED, "multi_device_energy", confidence, evidence
        )
    return IdentityDecision(attribution, "multi_device_energy", confidence, evidence)


def prepare_multi_device_energy(
    recording: FocalRecording,
    focal_audio: DecodedAudio,
    other_audio: dict[str, DecodedAudio],
    config: MultiDeviceEnergyConfig,
) -> MultiDeviceEnergyState:
    """Compute envelopes, align other devices to the focal one and validate on annotations."""
    frame_s = config.frame_s
    focal_db = energy_envelope_db(
        focal_audio.samples, focal_audio.sample_rate_hz, frame_s=frame_s
    )
    max_lag = round(config.sync_max_lag_s / frame_s)
    others: dict[str, np.ndarray] = {}
    lags: dict[str, int] = {}
    correlations: dict[str, float] = {}
    excluded: list[str] = []
    for view_id, audio in other_audio.items():
        envelope = energy_envelope_db(
            audio.samples, audio.sample_rate_hz, frame_s=frame_s
        )
        lag, correlation = envelope_lag_frames(
            focal_db, envelope, max_lag_frames=max_lag
        )
        lags[view_id], correlations[view_id] = lag, correlation
        if not math.isfinite(correlation) or correlation < config.sync_min_correlation:
            excluded.append(view_id)
            continue
        aligned = shift_envelope(envelope, lag)
        if len(aligned) < len(focal_db):
            aligned = np.concatenate(
                [aligned, np.full(len(focal_db) - len(aligned), np.nan)]
            )
        others[view_id] = aligned[: len(focal_db)]
    state = MultiDeviceEnergyState(
        frame_s=frame_s,
        canonical_offset_s=recording.canonical_start_s,
        focal_db=focal_db,
        others_db=others,
        lags_frames=lags,
        correlations=correlations,
        excluded_views=tuple(excluded),
    )
    _validate_on_annotations(state, recording, config)
    return state


def _validate_on_annotations(
    state: MultiDeviceEnergyState,
    recording: FocalRecording,
    config: MultiDeviceEnergyConfig,
) -> None:
    """Measure precision of the dominance rule on annotated focal/other intervals.

    Intervals and the state use the same canonical timeline. The state performs
    the conversion to envelope-local indices exactly once.
    """
    if not state.others_db:
        return
    focal = as_intervals(
        recording.focal_annotation_raw
        if recording.focal_annotation_raw is not None
        else recording.focal_annotation
    )
    other = as_intervals(recording.other_annotation)

    def dominances(intervals: Intervals) -> np.ndarray:
        values = [state.dominance_db(s, e) for s, e in intervals]
        return np.asarray([v for v in values if math.isfinite(v)], dtype=float)

    focal_dom, other_dom = dominances(focal), dominances(other)
    state.n_validation_focal, state.n_validation_other = len(focal_dom), len(other_dom)
    if (
        len(focal_dom) < config.validation_min_intervals
        or len(other_dom) < config.validation_min_intervals
    ):
        return
    threshold = config.dominance_threshold_db
    focal_as_focal = int((focal_dom >= threshold).sum())
    other_as_focal = int((other_dom >= threshold).sum())
    focal_as_other = int((focal_dom <= -threshold).sum())
    other_as_other = int((other_dom <= -threshold).sum())
    state.precision_focal = focal_as_focal / max(focal_as_focal + other_as_focal, 1)
    state.precision_other = other_as_other / max(other_as_other + focal_as_other, 1)
    state.recall_focal = focal_as_focal / len(focal_dom)
    both = np.concatenate([focal_dom, other_dom])
    state.undecided_ratio = float((np.abs(both) < threshold).mean())


@dataclass
class IdentityPipeline:
    """Ordered attribution methods for one dataset."""

    method_names: tuple[str, ...]
    config: IdentityConfig

    KNOWN: tuple[str, ...] = (
        "focal_annotation",
        "other_annotation",
        "multi_device_energy",
    )

    def __post_init__(self) -> None:
        unknown = set(self.method_names) - set(self.KNOWN)
        if unknown:
            raise ValueError(f"Unknown identity methods: {sorted(unknown)}")

    @property
    def name(self) -> str:
        """Pipeline description for reports."""
        return "+".join(self.method_names)

    def prepare(
        self,
        recording: FocalRecording,
        focal_audio: DecodedAudio | None,
        other_audio: dict[str, DecodedAudio],
    ) -> RecordingIdentityContext:
        """Per-recording preparation; only the energy method needs audio."""
        context = RecordingIdentityContext(method_names=self.method_names)
        if (
            "multi_device_energy" in self.method_names
            and focal_audio is not None
            and other_audio
        ):
            state = prepare_multi_device_energy(
                recording, focal_audio, other_audio, self.config.multi_device_energy
            )
            context.energy = state
            context.confidence = state.precision_focal
            context.can_attribute_unannotated_focal = bool(
                math.isfinite(state.precision_focal)
                and state.precision_focal
                >= self.config.multi_device_energy.min_identity_confidence
            )
            context.notes = {
                "precision_focal": state.precision_focal,
                "precision_other": state.precision_other,
                "recall_focal": state.recall_focal,
                "undecided_ratio": state.undecided_ratio,
                "n_validation_focal": float(state.n_validation_focal),
                "n_validation_other": float(state.n_validation_other),
                "n_synchronized_other_views": float(len(state.others_db)),
                "n_excluded_other_views": float(len(state.excluded_views)),
                **{
                    f"sync_lag_frames:{v}": float(lag)
                    for v, lag in state.lags_frames.items()
                },
                **{f"sync_correlation:{v}": c for v, c in state.correlations.items()},
            }
        return context

    def attribute(
        self, context: RecordingIdentityContext, segment: SegmentContext
    ) -> IdentityDecision:
        """First deciding method wins; otherwise unresolved."""
        for name in self.method_names:
            decision: IdentityDecision | None
            if name == "focal_annotation":
                decision = focal_annotation_method(segment, self.config)
            elif name == "other_annotation":
                decision = other_annotation_method(segment, self.config)
            else:
                decision = multi_device_energy_method(
                    segment, context.energy, self.config.multi_device_energy
                )
            if decision is not None:
                return decision
        return IdentityDecision(SpeakerAttribution.UNRESOLVED, "none", math.nan, {})
