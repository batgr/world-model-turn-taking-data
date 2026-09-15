"""Typed records produced by the audio timeline audit.

Records are dataclasses composed from smaller metric groups. ``to_row`` flattens a
record into the column layout written to Parquet; the field names *are* the
report schema, so renaming a field renames a column.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any


class TimelineStatus(StrEnum):
    """Whether a file yielded enough packets to measure its timeline."""

    MEASURED = "measured"
    INSUFFICIENT_DATA = "insufficient_data"
    PROBE_ERROR = "probe_error"


class EventCategory(StrEnum):
    """Diagnostic category of one audio timeline event.

    ``PACKET_TIMELINE_ERROR`` is a container inconsistency (PTS step differs from
    the packet's own duration). All other categories describe the PCM-vs-PTS
    clock: ``AUDIO_DROPOUT`` is a positive step whose excess persists beyond the
    compensation window; ``COMPENSATED_TIMESTAMP_CADENCE`` groups positive steps
    that surrounding short steps cancel locally; ``STITCH_RELATED`` and
    ``CODEC_BOUNDARY`` are dataset- or codec-interpreted relabels applied after
    generic classification; ``PCM_TIMESTAMP_VARIATION`` is everything else that
    exceeds the drift threshold.
    """

    PACKET_TIMELINE_ERROR = "packet_timeline_error"
    AUDIO_DROPOUT = "audio_dropout"
    COMPENSATED_TIMESTAMP_CADENCE = "compensated_timestamp_cadence"
    PCM_TIMESTAMP_VARIATION = "pcm_timestamp_variation"
    STITCH_RELATED = "stitch_related_pcm_discontinuity"
    CODEC_BOUNDARY = "codec_boundary_pcm_event"


class EventType(StrEnum):
    """Structural kind of an event row, independent of its interpretation."""

    PACKET_TIMELINE_GAP = "packet_timeline_gap"
    PACKET_TIMELINE_OVERLAP = "packet_timeline_overlap"
    PCM_STEP_EPISODE = "pcm_step_episode"
    AUDIO_DROPOUT = "audio_dropout"


class EventDirection(StrEnum):
    """Sign of the clock error inside an event."""

    GAP = "gap"
    OVERLAP = "overlap"
    MIXED = "mixed"


@dataclass(frozen=True)
class AudioTimelineParameters:
    """Settings that decide how PCM-vs-PTS measurements are classified.

    All threshold comparisons are strict (``value > threshold``).
    """

    reference_decoded_samples: int
    """Decoded samples per packet (1024 for AAC-LC), validated by decoding."""
    drift_threshold_ms: float = 1.0
    """Minimum PCM step error, in milliseconds, for an event to be reported."""
    compensation_window_sec: float = 2.0
    """How long a positive step must persist before it counts as a dropout."""
    persistence_fraction: float = 0.5
    """Fraction of the step excess that must survive the compensation window."""
    threshold_comparison: str = "strictly_greater"

    def __post_init__(self) -> None:
        if self.reference_decoded_samples <= 0:
            raise ValueError("reference_decoded_samples must be positive.")
        if self.drift_threshold_ms <= 0:
            raise ValueError("drift_threshold_ms must be positive.")
        if self.compensation_window_sec <= 0:
            raise ValueError("compensation_window_sec must be positive.")
        if not 0 < self.persistence_fraction <= 1:
            raise ValueError("persistence_fraction must be in (0, 1].")

    def threshold_samples(self, sample_rate_hz: int) -> float:
        """Drift threshold expressed in audio samples at ``sample_rate_hz``."""
        return sample_rate_hz * self.drift_threshold_ms / 1000.0

    def window_steps(self, sample_rate_hz: int) -> int:
        """Compensation window expressed in packets (at least one)."""
        return max(
            1,
            math.ceil(
                self.compensation_window_sec
                * sample_rate_hz
                / self.reference_decoded_samples
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class PacketCoverage:
    """How many packets carried the fields the analysis needs."""

    n_packets: int
    n_valid_packets: int
    """Packets with both a PTS and a duration."""
    timestamp_coverage: float
    duration_coverage: float
    timestamp_duration_coverage: float
    pts_monotonic: bool
    n_non_positive_pts_steps: int
    samples_per_time_base_tick_numerator: int
    samples_per_time_base_tick_denominator: int


@dataclass(frozen=True)
class PacketClockMetrics:
    """Container self-consistency: ``next_pts - pts - duration`` per packet."""

    n_valid_packet_timeline_steps: int
    packet_timeline_gap_count: int
    packet_timeline_overlap_count: int
    packet_max_abs_timeline_error_samples: float


@dataclass(frozen=True)
class PacketDurationMetrics:
    """Distribution of packet durations relative to the decoded reference.

    ``robust_nominal_packet_duration_samples`` is the per-file median; the long
    duration fence is ``max(median + max(6 * MAD, 0.5 * median), reference +
    threshold)``. Abnormally long packets are an observation only.
    """

    nominal_packet_duration_samples: float
    robust_nominal_packet_duration_samples: float
    packet_duration_mad_samples: float
    long_packet_duration_threshold_samples: float
    min_packet_duration_samples: float
    median_packet_duration_samples: float
    max_packet_duration_samples: float
    n_distinct_packet_durations: int
    n_packet_duration_variations: int
    max_abs_packet_duration_deviation_samples: float
    n_abnormally_long_packet_durations: int

    @classmethod
    def empty(cls) -> PacketDurationMetrics:
        """Metrics for a file whose packets carry no duration."""
        nan = math.nan
        return cls(nan, nan, nan, nan, nan, nan, nan, 0, 0, nan, 0)


@dataclass(frozen=True)
class PcmClockMetrics:
    """PCM-vs-PTS clock: ``next_pts - pts - reference_decoded_samples`` per step.

    Cumulative drift is a file property: ``final_*`` is the offset a naïve
    concatenation of decoded PCM would carry at the end of the file, and
    ``max_abs_*`` locates the largest transient offset in native PTS seconds.
    """

    n_valid_pts_steps: int
    n_pcm_step_variations: int
    max_abs_pcm_step_error_samples: float
    pcm_step_error_p95_abs_samples: float
    final_cumulative_pcm_drift_samples: float
    final_cumulative_pcm_drift_ms: float
    max_abs_cumulative_pcm_drift_samples: float
    max_abs_cumulative_pcm_drift_ms: float
    max_abs_cumulative_pcm_drift_time_sec: float
    """Native PTS time (seconds) at which the absolute cumulative drift peaks."""
    has_timestamp_jitter: bool
    has_small_bounded_timestamp_jitter: bool
    has_significant_transient_pcm_offset: bool
    has_significant_final_pcm_offset: bool


@dataclass(frozen=True)
class DropoutAccounting:
    """Dropout candidates split by persistence through the compensation window."""

    n_dropout_candidate_steps: int
    n_persistent_dropout_steps: int
    n_compensated_candidate_steps: int


@dataclass(frozen=True)
class CodecBoundaryMetadata:
    """FFmpeg ``Skip Samples`` side data summed over the stream."""

    skip_samples: int
    discard_padding_samples: int


@dataclass(frozen=True)
class AudioFileInterpretation:
    """File-level flags derived from interpreted events and final drift.

    Final-drift flags use strict comparisons; ``half_video_frame_ms`` is computed
    from the file's own video frame rate.
    """

    has_stitch_related_event: bool
    has_early_boundary_event: bool
    has_audio_dropout: bool
    audio_dropout_event_count: int
    audio_dropout_packet_count: int
    max_audio_dropout_duration_ms: float
    total_audio_dropout_excess_ms: float
    has_compensated_timestamp_cadence: bool
    abs_final_cumulative_pcm_drift_ms: float
    half_video_frame_ms: float
    final_pcm_drift_exceeds_1_ms: bool
    final_pcm_drift_exceeds_half_video_frame: bool
    final_pcm_drift_exceeds_100_ms: bool
    final_pcm_drift_exceeds_500_ms: bool

    @classmethod
    def empty(cls, *, video_fps: float) -> AudioFileInterpretation:
        """Flags for a file without events or without a measured timeline."""
        return cls(
            has_stitch_related_event=False,
            has_early_boundary_event=False,
            has_audio_dropout=False,
            audio_dropout_event_count=0,
            audio_dropout_packet_count=0,
            max_audio_dropout_duration_ms=0.0,
            total_audio_dropout_excess_ms=0.0,
            has_compensated_timestamp_cadence=False,
            abs_final_cumulative_pcm_drift_ms=math.nan,
            half_video_frame_ms=500.0 / video_fps if video_fps > 0 else math.nan,
            final_pcm_drift_exceeds_1_ms=False,
            final_pcm_drift_exceeds_half_video_frame=False,
            final_pcm_drift_exceeds_100_ms=False,
            final_pcm_drift_exceeds_500_ms=False,
        )


@dataclass(frozen=True)
class AudioFileTimelineRecord:
    """File-level result of the packet timeline audit (one Parquet row)."""

    dataset: str
    relative_path: str
    audio_codec: str
    sample_rate_hz: int
    time_base: str
    audio_duration_sec: float
    video_avg_frame_rate: float
    reference_decoded_samples_source: str
    probe_ok: bool
    probe_error: str | None
    timeline_status: TimelineStatus
    parameters: AudioTimelineParameters
    coverage: PacketCoverage | None = None
    packet_clock: PacketClockMetrics | None = None
    durations: PacketDurationMetrics | None = None
    pcm_clock: PcmClockMetrics | None = None
    dropouts: DropoutAccounting | None = None
    codec_boundary: CodecBoundaryMetadata | None = None
    compensation_window_steps: int | None = None
    interpretation: AudioFileInterpretation | None = None
    """Flags added after generic analysis by the interpretation step."""

    @classmethod
    def failed(
        cls,
        *,
        dataset: str,
        relative_path: str,
        audio_codec: str,
        sample_rate_hz: int,
        time_base: str,
        audio_duration_sec: float,
        video_avg_frame_rate: float,
        reference_decoded_samples_source: str,
        parameters: AudioTimelineParameters,
        error: str,
    ) -> AudioFileTimelineRecord:
        """Record for a file whose probe or analysis raised."""
        return cls(
            dataset=dataset,
            relative_path=relative_path,
            audio_codec=audio_codec,
            sample_rate_hz=sample_rate_hz,
            time_base=time_base,
            audio_duration_sec=audio_duration_sec,
            video_avg_frame_rate=video_avg_frame_rate,
            reference_decoded_samples_source=reference_decoded_samples_source,
            probe_ok=False,
            probe_error=error,
            timeline_status=TimelineStatus.PROBE_ERROR,
            parameters=parameters,
        )

    def to_row(self) -> dict[str, Any]:
        """Flatten into the Parquet column layout."""
        row: dict[str, Any] = {
            "dataset": self.dataset,
            "relative_path": self.relative_path,
            "audio_codec": self.audio_codec,
            "sample_rate_hz": self.sample_rate_hz,
            "time_base": self.time_base,
            "audio_duration_sec": self.audio_duration_sec,
            "video_avg_frame_rate": self.video_avg_frame_rate,
            "reference_decoded_samples_source": self.reference_decoded_samples_source,
            "probe_ok": self.probe_ok,
            "probe_error": self.probe_error,
            "timeline_status": str(self.timeline_status),
            "reference_decoded_samples": self.parameters.reference_decoded_samples,
            "drift_threshold_ms": self.parameters.drift_threshold_ms,
            "drift_threshold_samples": self.parameters.threshold_samples(
                self.sample_rate_hz
            ),
            "threshold_comparison": self.parameters.threshold_comparison,
            "compensation_window_sec": self.parameters.compensation_window_sec,
            "compensation_window_steps": self.compensation_window_steps,
            "persistence_fraction": self.parameters.persistence_fraction,
        }
        for group in (
            self.coverage,
            self.packet_clock,
            self.durations,
            self.pcm_clock,
            self.dropouts,
            self.codec_boundary,
            self.interpretation,
        ):
            if group is not None:
                row.update(asdict(group))
        return row


@dataclass(frozen=True)
class AudioTimelineEvent:
    """One local event on an audio timeline (one Parquet row).

    ``packet_index`` is the packet *before* the step that starts the event and
    ``event_pts`` the PTS of the packet after it. Dropout fields are ``NaN`` for
    non-dropout rows.
    """

    packet_index: int
    next_packet_index: int
    event_type: EventType
    event_category: EventCategory
    event_direction: EventDirection
    event_pts: int
    event_time_sec: float
    event_end_time_sec: float
    n_steps: int
    magnitude_samples: float
    packet_timeline_error_samples: float
    first_pcm_step_error_samples: float
    max_abs_pcm_step_error_samples: float
    net_pcm_drift_samples: float
    robust_nominal_packet_duration_samples: float
    long_packet_duration_threshold_samples: float
    n_abnormally_long_packets: int
    max_packet_duration_samples: float
    n_candidate_steps: int
    n_compensated_candidate_steps: int
    dropout_duration_samples: float
    dropout_duration_ms: float
    dropout_total_excess_samples: float
    dropout_total_excess_ms: float
    dropout_min_residual_in_window_samples: float
    """Lowest drift relative to the pre-step baseline inside the window."""
    dropout_residual_at_window_end_samples: float
    dropout_window_truncated: bool
    """``True`` when the stream ended before the window did."""
    reference_decoded_samples: int

    def to_row(self) -> dict[str, Any]:
        """Flatten into the Parquet column layout."""
        row = asdict(self)
        row["event_type"] = str(self.event_type)
        row["event_category"] = str(self.event_category)
        row["event_direction"] = str(self.event_direction)
        return row

    def relabel(self, category: EventCategory) -> AudioTimelineEvent:
        """Return a copy with an interpreted category."""
        return replace(self, event_category=category)


EVENT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(AudioTimelineEvent))
"""Event column order as written by the generic analysis."""


@dataclass(frozen=True)
class DecodedValidationRecord:
    """Decoded ``-show_frames`` evidence for one selected window."""

    case: str
    selection_reason: str
    dataset: str
    relative_path: str
    requested_interval: str
    event_time_sec: float | None
    audio_codec: str
    sample_rate_hz: int
    time_base: str
    packet_nominal_duration_samples: float
    robust_nominal_packet_duration_samples: float
    long_packet_duration_threshold_samples: float
    packet_max_abs_duration_deviation_samples: float
    skip_samples: int
    discard_padding_samples: int
    probe_ok: bool
    probe_error: str | None
    n_decoded_frames: int = 0
    n_valid_decoded_sample_counts: int = 0
    reference_decoded_samples: int | None = None
    reference_method: str = "mode_nb_samples"
    n_non_reference_decoded_frames: int = 0
    n_960_sample_frames: int = 0
    decoded_sample_count_distribution_json: str = "{}"
    reference_matches_mode: bool = False
    decoded_pts_max_abs_step_error_samples: float = math.nan

    def to_row(self) -> dict[str, Any]:
        """Flatten into the Parquet column layout."""
        return asdict(self)


def ticks_to_seconds(pts: int, time_base: Fraction) -> float:
    """Convert an integer PTS to seconds through the exact rational time base."""
    return float(Fraction(pts) * time_base)
