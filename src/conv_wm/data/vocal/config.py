"""Versioned configuration of the vocal annotation coverage audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from conv_wm.config import PROJECT_ROOT

DEFAULT_COVERAGE_CONFIG_PATH = PROJECT_ROOT / "conf" / "vocal_annotation_coverage.yaml"


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters of the acoustic voice activity detector."""

    name: str = "silero_vad"
    sample_rate_hz: int = 16000
    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    window_size_samples: int = 512


@dataclass(frozen=True)
class CoverageConfig:
    """How detected segments are compared with focal annotations."""

    annotation_overlap_tolerance_s: float = 0.2
    covered_min_overlap_ratio: float = 0.9
    uncovered_max_overlap_ratio: float = 0.1
    annotation_merge_gap_s: float = 0.3
    duration_bucket_edges_s: tuple[float, ...] = (0.25, 0.5, 1.0)
    short_segment_max_duration_s: float = 0.5

    def __post_init__(self) -> None:
        if (
            not 0
            <= self.uncovered_max_overlap_ratio
            < self.covered_min_overlap_ratio
            <= 1
        ):
            raise ValueError(
                "overlap ratio thresholds must satisfy 0 <= uncovered < covered <= 1"
            )
        if list(self.duration_bucket_edges_s) != sorted(self.duration_bucket_edges_s):
            raise ValueError("duration_bucket_edges_s must be sorted")

    def bucket_labels(self) -> tuple[str, ...]:
        """Human-readable labels of the duration buckets, in order."""
        edges = self.duration_bucket_edges_s
        labels = [f"<{edges[0]}s"]
        labels += [f"{lo}-{hi}s" for lo, hi in pairwise(edges)]
        labels.append(f">={edges[-1]}s")
        return tuple(labels)

    def bucket_of(self, duration_s: float) -> str:
        """Label of the bucket containing ``duration_s``."""
        labels = self.bucket_labels()
        for edge, label in zip(self.duration_bucket_edges_s, labels[:-1], strict=True):
            if duration_s < edge:
                return label
        return labels[-1]


@dataclass(frozen=True)
class MultiDeviceEnergyConfig:
    """Parameters of the synchronized multi-device energy dominance identity method."""

    frame_s: float = 0.01
    dominance_threshold_db: float = 3.0
    min_identity_confidence: float = 0.8
    validation_min_intervals: int = 20
    sync_max_lag_s: float = 2.0
    sync_min_correlation: float = 0.2


@dataclass(frozen=True)
class IdentityConfig:
    """Parameters of the speaker attribution methods."""

    focal_annotation_min_overlap_ratio: float = 0.5
    other_annotation_min_overlap_ratio: float = 0.5
    multi_device_energy: MultiDeviceEnergyConfig = field(
        default_factory=MultiDeviceEnergyConfig
    )

    def __post_init__(self) -> None:
        for name, value in (
            (
                "focal_annotation_min_overlap_ratio",
                self.focal_annotation_min_overlap_ratio,
            ),
            (
                "other_annotation_min_overlap_ratio",
                self.other_annotation_min_overlap_ratio,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"identity.{name} must be in [0, 1]")


@dataclass(frozen=True)
class VocalCoverageConfig:
    """Complete, versioned configuration of one audit run."""

    schema_version: int = 1
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    source_path: str | None = None
    """Where the values were loaded from; ``None`` for in-code defaults."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping (tuples as lists, NaN/inf never present)."""
        return json.loads(json.dumps(asdict(self)))

    def checksum(self) -> str:
        """SHA-256 of the canonical JSON form, independent of ``source_path``."""
        payload = self.to_dict()
        payload.pop("source_path", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_coverage_config(path: Path | None = None) -> VocalCoverageConfig:
    """Load the YAML configuration, validating every field against the dataclasses."""
    path = path or DEFAULT_COVERAGE_CONFIG_PATH
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"{path}: configuration root must be a mapping")
    raw: dict[str, Any] = {str(key): value for key, value in loaded.items()}
    return _from_mapping(raw, source_path=str(path))


def _from_mapping(
    raw: dict[str, Any], *, source_path: str | None
) -> VocalCoverageConfig:
    detector = raw.get("detector", {})
    coverage = dict(raw.get("coverage", {}))
    identity = dict(raw.get("identity", {}))
    if "duration_bucket_edges_s" in coverage:
        coverage["duration_bucket_edges_s"] = tuple(
            float(v) for v in coverage["duration_bucket_edges_s"]
        )
    energy = identity.pop("multi_device_energy", {})
    config = VocalCoverageConfig(
        schema_version=int(raw.get("schema_version", 1)),
        detector=DetectorConfig(**detector),
        coverage=CoverageConfig(**coverage),
        identity=IdentityConfig(
            **identity, multi_device_energy=MultiDeviceEnergyConfig(**energy)
        ),
        source_path=source_path,
    )
    for name, value in asdict(config.detector).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"detector.{name} must be finite")
    return config
