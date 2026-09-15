"""Packet-level audio timeline analysis of one stream."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from conv_wm.data.media.audio.packets import (
    PacketTimeline,
    packet_clock_errors,
    packet_duration_metrics,
    step_duration_samples,
)
from conv_wm.data.media.audio.pcm_clock import (
    candidate_persistence,
    packet_clock_events,
    pcm_clock_events,
    pcm_clock_metrics,
    pcm_clock_series,
)
from conv_wm.data.media.audio.probe import extract_skip_samples
from conv_wm.data.media.audio.records import (
    EVENT_COLUMNS,
    AudioFileTimelineRecord,
    AudioTimelineEvent,
    CodecBoundaryMetadata,
    TimelineStatus,
)


@dataclass(frozen=True)
class AudioPacketAnalysis:
    """Result of analysing one stream: a file record and its local events."""

    file: AudioFileTimelineRecord
    events: list[AudioTimelineEvent]

    def events_frame(self) -> pd.DataFrame:
        """Events as a DataFrame with the canonical column order."""
        if not self.events:
            return pd.DataFrame(columns=list(EVENT_COLUMNS))
        return pd.DataFrame.from_records([event.to_row() for event in self.events])


def codec_boundary_metadata(packets: list[dict[str, object]]) -> CodecBoundaryMetadata:
    """Sum ``Skip Samples`` side data over the stream."""
    skip_data = extract_skip_samples(packets)
    if skip_data.empty:
        return CodecBoundaryMetadata(skip_samples=0, discard_padding_samples=0)
    return CodecBoundaryMetadata(
        skip_samples=int(skip_data["skip_samples"].sum()),
        discard_padding_samples=int(skip_data["discard_padding"].sum()),
    )


def analyze_audio_packets(
    packets: list[dict[str, object]],
    *,
    file: AudioFileTimelineRecord,
) -> AudioPacketAnalysis:
    """Measure the packet clock and the PCM-vs-PTS clock of one audio stream.

    ``file`` carries the stream identity and the analysis parameters; the
    returned record is a copy with every metric group filled in. Streams with
    fewer than two packets are reported as ``INSUFFICIENT_DATA``.
    """
    parameters = file.parameters
    timeline = PacketTimeline.from_packets(
        packets,
        sample_rate_hz=file.sample_rate_hz,
        time_base=file.time_base,
    )
    coverage = timeline.coverage()
    boundary = codec_boundary_metadata(packets)
    window_steps = parameters.window_steps(file.sample_rate_hz)
    if timeline.n_packets < 2:
        record = AudioFileTimelineRecord(
            **{
                **_identity(file),
                "timeline_status": TimelineStatus.INSUFFICIENT_DATA,
                "coverage": coverage,
                "codec_boundary": boundary,
                "compensation_window_steps": window_steps,
            }
        )
        return AudioPacketAnalysis(file=record, events=[])

    threshold = parameters.threshold_samples(file.sample_rate_hz)
    clock_errors = packet_clock_errors(timeline)
    durations = packet_duration_metrics(
        timeline,
        reference_decoded_samples=parameters.reference_decoded_samples,
        threshold_samples=threshold,
    )
    series = pcm_clock_series(
        timeline,
        reference_decoded_samples=parameters.reference_decoded_samples,
        step_duration_samples=step_duration_samples(timeline),
    )
    persistence = candidate_persistence(
        series,
        threshold_samples=threshold,
        window_steps=window_steps,
        persistence_fraction=parameters.persistence_fraction,
    )
    record = AudioFileTimelineRecord(
        **{
            **_identity(file),
            "timeline_status": TimelineStatus.MEASURED,
            "coverage": coverage,
            "packet_clock": clock_errors.metrics(),
            "durations": durations,
            "pcm_clock": pcm_clock_metrics(
                series,
                time_base=timeline.time_base,
                sample_rate_hz=file.sample_rate_hz,
                threshold_samples=threshold,
            ),
            "dropouts": persistence.accounting(),
            "codec_boundary": boundary,
            "compensation_window_steps": window_steps,
        }
    )
    events = packet_clock_events(
        clock_errors,
        timeline,
        durations=durations,
        reference_decoded_samples=parameters.reference_decoded_samples,
    )
    events += pcm_clock_events(
        series,
        persistence,
        time_base=timeline.time_base,
        sample_rate_hz=file.sample_rate_hz,
        parameters=parameters,
        durations=durations,
    )
    return AudioPacketAnalysis(file=record, events=events)


def _identity(file: AudioFileTimelineRecord) -> dict[str, object]:
    """Identity fields shared by every record derived from ``file``."""
    return {
        "dataset": file.dataset,
        "relative_path": file.relative_path,
        "audio_codec": file.audio_codec,
        "sample_rate_hz": file.sample_rate_hz,
        "time_base": file.time_base,
        "audio_duration_sec": file.audio_duration_sec,
        "video_avg_frame_rate": file.video_avg_frame_rate,
        "reference_decoded_samples_source": file.reference_decoded_samples_source,
        "probe_ok": True,
        "probe_error": None,
        "parameters": file.parameters,
    }
