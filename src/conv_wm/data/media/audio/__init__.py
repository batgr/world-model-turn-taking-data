"""Audio timeline audit: packet clock, PCM-vs-PTS clock and decoded evidence.

Modules, in analysis order:

- ``probe``: ffprobe access (packets, decoded frames, side data);
- ``packets``: packet normalization, packet-clock and duration metrics;
- ``pcm_clock``: PCM-vs-PTS step errors, persistence, episode events;
- ``analysis``: one-stream orchestration into typed records;
- ``interpretation``: codec-boundary and known-boundary relabelling;
- ``decode``: decoded-frame validation of selected windows;
- ``records``: the typed record and enum vocabulary.
"""

from conv_wm.data.media.audio.analysis import AudioPacketAnalysis, analyze_audio_packets
from conv_wm.data.media.audio.decode import (
    DecodeValidationCase,
    decoded_frame_continuity,
    effective_audio_bounds,
    summarize_decoded_sample_counts,
    validate_decoded_window,
)
from conv_wm.data.media.audio.interpretation import (
    INTERPRETED_EVENT_COLUMNS,
    InterpretedAudioEvent,
    KnownBoundaryGrid,
    final_drift_flags,
    interpret_events,
    interpret_file,
    interpreted_events_frame,
)
from conv_wm.data.media.audio.probe import (
    extract_skip_samples,
    probe_audio_frames,
    probe_audio_packets,
)
from conv_wm.data.media.audio.records import (
    EVENT_COLUMNS,
    AudioFileInterpretation,
    AudioFileTimelineRecord,
    AudioTimelineEvent,
    AudioTimelineParameters,
    DecodedValidationRecord,
    EventCategory,
    EventDirection,
    EventType,
    TimelineStatus,
)

__all__ = [
    "EVENT_COLUMNS",
    "INTERPRETED_EVENT_COLUMNS",
    "AudioFileInterpretation",
    "AudioFileTimelineRecord",
    "AudioPacketAnalysis",
    "AudioTimelineEvent",
    "AudioTimelineParameters",
    "DecodeValidationCase",
    "DecodedValidationRecord",
    "EventCategory",
    "EventDirection",
    "EventType",
    "InterpretedAudioEvent",
    "KnownBoundaryGrid",
    "TimelineStatus",
    "analyze_audio_packets",
    "decoded_frame_continuity",
    "effective_audio_bounds",
    "extract_skip_samples",
    "final_drift_flags",
    "interpret_events",
    "interpret_file",
    "interpreted_events_frame",
    "probe_audio_frames",
    "probe_audio_packets",
    "summarize_decoded_sample_counts",
    "validate_decoded_window",
]
