"""Acoustic voice activity detectors.

A detector turns decoded audio into :class:`DetectedSegment`s on the canonical
timeline. The Silero detector is the production choice; ``StaticDetector``
replays pre-computed segments so the coverage logic can be tested without audio.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from conv_wm.data.vocal.audio import DecodedAudio
from conv_wm.data.vocal.config import DetectorConfig
from conv_wm.data.vocal.records import DetectedSegment


class VoiceActivityDetector(Protocol):
    """Detector contract used by the audit."""

    name: str

    def version(self) -> str:
        """Exact, reproducible identification of the detector (model + package)."""
        ...

    def detect(self, audio: DecodedAudio) -> list[DetectedSegment]:
        """Voice activity segments on the canonical timeline of ``audio``."""
        ...


@dataclass
class StaticDetector:
    """Replays segments given per view id; for tests and offline replays."""

    segments_by_view: Mapping[str, Sequence[DetectedSegment]]
    name: str = "static"

    def version(self) -> str:
        """Static detectors have no model; the version names the fixture."""
        return "static-fixture"

    def detect(self, audio: DecodedAudio) -> list[DetectedSegment]:
        """Segments registered for the audio's view, already on the canonical timeline."""
        return list(self.segments_by_view.get(audio.source.view_id, ()))


@dataclass
class SileroVoiceActivityDetector:
    """Silero VAD run on onnxruntime (CPU, single-threaded, deterministic)."""

    config: DetectorConfig
    name: str = "silero_vad"
    _model: object | None = None
    _timestamps: Callable[..., list[dict[str, float]]] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError(
                "Silero VAD is not installed; install the 'vad' extra: uv sync --extra vad"
            ) from exc
        self._model = load_silero_vad(onnx=True)
        self._timestamps = get_speech_timestamps

    def version(self) -> str:
        """``silero-vad <package version> onnx:<sha256 prefix of the model file>``."""
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import silero_vad

        model_file = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
        digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        return (
            f"silero-vad {metadata.version('silero-vad')} "
            f"onnxruntime {metadata.version('onnxruntime')} model_sha256:{digest}"
        )

    def detect(self, audio: DecodedAudio) -> list[DetectedSegment]:
        """Run the model and attach the mean chunk probability to each segment."""
        import torch

        self._load()
        assert self._model is not None and self._timestamps is not None
        if audio.sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError(
                f"detector expects {self.config.sample_rate_hz} Hz, got {audio.sample_rate_hz} Hz"
            )
        if len(audio.samples) == 0:
            return []
        waveform = torch.from_numpy(
            np.array(audio.samples, dtype=np.float32, order="C", copy=True)
        )
        window = self.config.window_size_samples
        self._model.reset_states()  # type: ignore[attr-defined]
        probabilities = (
            self._model.audio_forward(waveform, audio.sample_rate_hz)  # type: ignore[attr-defined]
            .numpy()
            .reshape(-1)
        )
        self._model.reset_states()  # type: ignore[attr-defined]
        kwargs: dict[str, object] = {
            "threshold": self.config.threshold,
            "sampling_rate": audio.sample_rate_hz,
            "min_speech_duration_ms": self.config.min_speech_duration_ms,
            "min_silence_duration_ms": self.config.min_silence_duration_ms,
            "speech_pad_ms": self.config.speech_pad_ms,
            "window_size_samples": window,
            "return_seconds": False,
        }
        if self.config.neg_threshold is not None:
            kwargs["neg_threshold"] = self.config.neg_threshold
        stamps = self._timestamps(waveform, self._model, **kwargs)
        segments: list[DetectedSegment] = []
        for stamp in stamps:
            start, end = int(stamp["start"]), int(stamp["end"])
            first, last = start // window, max(start // window, (end - 1) // window)
            chunk_probabilities = probabilities[first : last + 1]
            segments.append(
                DetectedSegment(
                    start_s=audio.canonical_time(start),
                    end_s=audio.canonical_time(end),
                    detection_confidence=(
                        float(chunk_probabilities.mean())
                        if len(chunk_probabilities)
                        else float("nan")
                    ),
                )
            )
        return segments


def build_detector(config: DetectorConfig) -> VoiceActivityDetector:
    """Instantiate the detector named in the configuration."""
    if config.name == "silero_vad":
        return SileroVoiceActivityDetector(config)
    raise ValueError(f"Unknown detector: {config.name}")
