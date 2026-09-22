import io

import numpy as np
from helpers import make_recording, seg

from conv_wm.data.audits.vocal_annotation_coverage import audit_recording_sets
from conv_wm.data.vocal.audio import DecodedAudio
from conv_wm.data.vocal.config import VocalCoverageConfig
from conv_wm.data.vocal.detector import StaticDetector
from conv_wm.data.vocal.records import (
    CoverageStatus,
    FocalVoiceActivityStatus,
)
from conv_wm.data.vocal.sources import DatasetRecordingSet


def test_annotation_only_population_stays_focal_unresolved():
    recording = make_recording(focal=[(1.0, 2.0)])
    source = DatasetRecordingSet(
        dataset="synthetic",
        recordings=[recording],
        annotation_paths=(),
        identity_methods=("focal_annotation", "other_annotation"),
        annotation_provenance="synthetic",
        limitations=(),
    )
    detector = StaticDetector({recording.view_id: [seg(1.0, 2.0), seg(4.0, 5.0)]})

    def decoder(audio_source, *, sample_rate_hz):
        return DecodedAudio(
            np.zeros(round(audio_source.duration_s * sample_rate_hz), dtype=np.float32),
            sample_rate_hz,
            audio_source,
        )

    [result] = audit_recording_sets(
        [source],
        detector=detector,
        decoder=decoder,
        config=VocalCoverageConfig(),
        stream=io.StringIO(),
    )

    assert [segment.coverage_status for segment in result.segments] == [
        CoverageStatus.COVERED,
        CoverageStatus.UNRESOLVED_IDENTITY,
    ]
    assert (
        result.summary.focal_voice_activity_status
        == FocalVoiceActivityStatus.UNRESOLVED
    )
