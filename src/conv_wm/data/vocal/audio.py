"""PTS-preserving audio decoding and short-time energy envelopes."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from conv_wm.data.media.ffprobe import FFprobeError
from conv_wm.data.vocal.records import AudioSource


def get_ffmpeg_path() -> str:
    """Return the ffmpeg executable available on PATH."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFprobeError("ffmpeg executable not found in PATH")
    return path


@dataclass(frozen=True)
class DecodedAudio:
    """Mono float32 samples whose index maps to native PTS.

    Decoding uses ``aresample=async=1:first_pts=0``: ffmpeg inserts silence at
    PTS gaps and drops samples at overlaps so that ``sample / sample_rate_hz``
    equals the stream's native time relative to the requested start. The
    requested window and the decoded length are both kept so the mapping can be
    checked (``duration_error_s``).
    """

    samples: np.ndarray
    sample_rate_hz: int
    source: AudioSource

    @property
    def duration_s(self) -> float:
        """Decoded length in seconds."""
        return len(self.samples) / self.sample_rate_hz

    @property
    def duration_error_s(self) -> float:
        """Decoded length minus the requested window length."""
        return self.duration_s - self.source.duration_s

    def canonical_time(self, sample_index: float) -> float:
        """Canonical time of a sample index."""
        return self.source.canonical_offset_s + sample_index / self.sample_rate_hz


def decode_audio(source: AudioSource, *, sample_rate_hz: int) -> DecodedAudio:
    """Decode one window of the first audio stream to mono PCM at ``sample_rate_hz``."""
    command = [
        get_ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{source.start_s:.6f}",
        "-i",
        source.path,
        "-t",
        f"{source.duration_s:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-af",
        "aresample=async=1:first_pts=0",
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise FFprobeError(
            f"ffmpeg failed for {source.path}: {result.stderr.decode(errors='replace').strip()}"
        )
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return DecodedAudio(samples=samples, sample_rate_hz=sample_rate_hz, source=source)


def energy_envelope_db(
    samples: np.ndarray, sample_rate_hz: int, *, frame_s: float
) -> np.ndarray:
    """RMS energy per ``frame_s`` frame, in dB (floor at -100 dB)."""
    frame = max(1, round(frame_s * sample_rate_hz))
    count = len(samples) // frame
    if count == 0:
        return np.empty(0, dtype=float)
    frames = samples[: count * frame].astype(np.float64).reshape(count, frame)
    rms = np.sqrt((frames**2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-5))


def envelope_lag_frames(
    reference: np.ndarray, other: np.ndarray, *, max_lag_frames: int
) -> tuple[int, float]:
    """Lag (frames) maximising the normalised cross-correlation of two envelopes.

    Positive lag means ``other`` is delayed relative to ``reference``. Returns
    ``(lag, correlation)``; correlation is NaN when an envelope is constant.
    """
    n = min(len(reference), len(other))
    x = reference[:n] - reference[:n].mean()
    y = other[:n] - other[:n].mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0 or n == 0:
        return 0, float("nan")
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        # ``other`` delayed by ``lag`` frames: other[t + lag] lines up with reference[t].
        if lag >= 0:
            value = float(np.dot(x[: n - lag], y[lag:]))
        else:
            value = float(np.dot(x[-lag:], y[: n + lag]))
        if value > best:
            best, best_lag = value, lag
    return best_lag, float(best / norm)


def shift_envelope(envelope: np.ndarray, lag_frames: int) -> np.ndarray:
    """Shift ``envelope`` so that it aligns with a reference it lags by ``lag_frames``."""
    if lag_frames == 0:
        return envelope
    shifted = np.full_like(envelope, np.nan)
    if lag_frames > 0:
        shifted[: len(envelope) - lag_frames] = envelope[lag_frames:]
    else:
        shifted[-lag_frames:] = envelope[: len(envelope) + lag_frames]
    return shifted
