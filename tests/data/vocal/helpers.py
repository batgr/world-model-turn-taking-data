import math

import numpy as np

from conv_wm.data.vocal.config import (
    CoverageConfig,
    IdentityConfig,
    VocalCoverageConfig,
)
from conv_wm.data.vocal.coverage import assess_recording
from conv_wm.data.vocal.identity import RecordingIdentityContext
from conv_wm.data.vocal.records import DetectedSegment, FocalRecording


def make_recording(
    focal=(),
    other=(),
    *,
    annotation_available=True,
    duration_s=60.0,
    recording_id="rec-1",
) -> FocalRecording:
    from conv_wm.data.vocal.records import AudioSource

    return FocalRecording(
        dataset="synthetic",
        recording_id=recording_id,
        view_id="view-1",
        wearer_id="w1",
        sync_group_id="group-1",
        audio=AudioSource("unused.mp4", 0.0, duration_s, 0.0, "view-1"),
        canonical_start_s=0.0,
        canonical_end_s=duration_s,
        focal_annotation=np.array(focal, dtype=float).reshape(-1, 2),
        other_annotation=np.array(other, dtype=float).reshape(-1, 2),
        annotation_available=annotation_available,
        annotation_source="synthetic",
    )


def seg(start: float, end: float, confidence: float = 0.9) -> DetectedSegment:
    return DetectedSegment(start, end, confidence)


def assess(recording, segments, pipeline, config: VocalCoverageConfig, *, context=None):
    context = context or RecordingIdentityContext(pipeline.method_names)
    return assess_recording(
        recording,
        segments,
        attribute=lambda segment: pipeline.attribute(context, segment),
        config=config.coverage,
        detection_method="static",
        identity_method=pipeline.name,
        identity_confidence=context.confidence,
    )


__all__ = [
    "CoverageConfig",
    "IdentityConfig",
    "assess",
    "make_recording",
    "math",
    "seg",
]
