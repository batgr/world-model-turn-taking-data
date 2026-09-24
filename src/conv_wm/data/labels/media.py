"""The media extractors: decoded audio / video -> diagnostic and prosodic grid labels.

Only these extractors open media files, and each decodes only its own stream:
the audio extractor never decodes video frames and the video extractor never
decodes audio. Files are resolved through the media manifest built from the
same action grid (a manifest from another grid is refused) and read from the
local corpus root ``${paths.raw}/<corpus_directory>``; nothing is downloaded.

Audio labels on the mixed signal are diagnostic controls. The wearer's
prosody is computed only on samples where the native annotations say the
wearer is the sole speaker (the *ego-solo mask*): overlap and other speakers
are always masked, and a cell with too little masked time is null.
``prosody.ego_f0_hz`` and ``prosody.ego_voiced_fraction`` come from Praat
(through Parselmouth, the ``labels-audio`` extra) and are built only on
request; their tool versions and parameters are recorded in the manifest.

Decoding uses FFmpeg with input seeking to ``t_k + media_offset_s``; audio is
resampled to ``audio_sample_rate_hz`` mono, video is resampled to the subframe
rate (one frame per subframe) and scaled to ``video_frame_width`` pixels.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.labels.catalog import PRAAT
from conv_wm.data.labels.gridding import Cumulative, scalar
from conv_wm.data.labels.registry import Extractor, Table
from conv_wm.data.labels.speech import GridKeys, key_columns
from conv_wm.data.labels.timeline import (
    EPSILON_S,
    SPEAKING,
    LabelConfig,
    RecordingStructure,
)
from conv_wm.data.vocal.action_grid import DECISION_STEP_S

SILENCE_DB = -120.0
"""dBFS assigned to a digital-silence RMS of zero before masking (never stored)."""
DOMINANT_LEVELS = 4
"""Quantization levels per RGB channel for the dominant colour."""


class MissingOptionalDependencyError(ImportError):
    """An extractor needs an optional dependency that is not installed."""


def require_module(module: str, *, label: str, extra: str) -> Any:
    """Import ``module`` or explain which extra installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingOptionalDependencyError(
            f"{label} needs the optional dependency {module!r}: install it with "
            f"`uv sync --extra {extra}`"
        ) from error


def require_ffmpeg() -> None:
    """Refuse to decode without FFmpeg on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MissingPrerequisiteError(
                Path(tool), produce_with="install FFmpeg (ffmpeg and ffprobe on PATH)"
            )


# -- decoding ---------------------------------------------------------------


def decode_audio(
    path: Path, start_s: float, duration_s: float, sample_rate: int
) -> np.ndarray:
    """Mono float32 PCM of ``[start_s, start_s + duration_s)`` of a media file."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{max(start_s, 0.0):.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def probe_size(path: Path) -> tuple[int, int]:
    """Width and height of the first video stream."""
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    width, height = (int(value) for value in output.split(",")[:2])
    return width, height


def iter_video_frames(
    path: Path, start_s: float, duration_s: float, rate_hz: float, width: int
) -> Iterator[np.ndarray]:
    """RGB frames ``(h, w, 3)`` resampled to ``rate_hz``, streamed (bounded memory)."""
    source_width, source_height = probe_size(path)
    height = max(2, round(source_height * width / source_width / 2) * 2)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{max(start_s, 0.0):.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(path),
        "-an",
        "-vf",
        f"fps={rate_hz:.6f},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    size = width * height * 3
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as process:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(size)
            if len(chunk) < size:
                break
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)


# -- pure features ----------------------------------------------------------


def audio_cell_features(
    samples: np.ndarray, sample_rate: int, cells: int, step_s: float
) -> dict[str, np.ndarray]:
    """Mixed-signal controls per cell; ``audio_valid`` is false for incomplete cells."""
    per_cell = round(sample_rate * step_s)
    available = min(cells, len(samples) // per_cell)
    valid = np.zeros(cells, dtype=bool)
    valid[:available] = True
    frames = np.zeros((cells, per_cell), dtype=np.float64)
    if available:
        frames[:available] = samples[: available * per_cell].reshape(
            available, per_cell
        )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    window = np.hanning(per_cell)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    frequencies = np.fft.rfftfreq(per_cell, 1.0 / sample_rate)
    energy = magnitude.sum(axis=1)
    spectral = valid & (energy > 0)
    safe = np.where(energy > 0, energy, 1.0)
    centroid = (magnitude * frequencies).sum(axis=1) / safe
    bandwidth = np.sqrt(
        (magnitude * (frequencies - centroid[:, None]) ** 2).sum(axis=1) / safe
    )
    normalized = magnitude / safe[:, None]
    flux = np.full(cells, np.nan)
    if cells > 1:
        flux[1:] = np.linalg.norm(normalized[1:] - normalized[:-1], axis=1)
    flux_valid = spectral & np.concatenate([[False], spectral[:-1]])
    return {
        "global_audio_rms": np.where(valid, rms, np.nan),
        "zero_crossing_rate": np.where(valid, zcr, np.nan),
        "spectral_centroid": np.where(spectral, centroid, np.nan),
        "spectral_bandwidth": np.where(spectral, bandwidth, np.nan),
        "spectral_flux": np.where(flux_valid, flux, np.nan),
        "spectral_valid": spectral,
        "background_noise_proxy": np.where(
            valid, np.quantile(np.abs(frames), 0.2, axis=1), np.nan
        ),
        "audio_valid": valid,
    }


def ego_solo_mask(structure: RecordingStructure, times: np.ndarray) -> np.ndarray:
    """Whether the wearer is the only (known) speaker at each time."""
    pieces = structure.pieces
    solo = (
        pieces.all_known
        & (pieces.states[:, 0] == SPEAKING)
        & (pieces.speaking_count == 1)
    )
    index = (
        np.searchsorted(pieces.starts, np.asarray(times) + EPSILON_S, side="right") - 1
    )
    inside = (index >= 0) & (np.asarray(times) < pieces.ends[-1])
    return inside & solo[np.clip(index, 0, None)]


def ego_solo_fraction(
    structure: RecordingStructure, starts: np.ndarray, step_s: float
) -> np.ndarray:
    """Fraction of each cell covered by the ego-solo mask (NaN where anything is UNKNOWN)."""
    pieces = structure.pieces
    solo = (
        pieces.all_known
        & (pieces.states[:, 0] == SPEAKING)
        & (pieces.speaking_count == 1)
    )
    fraction = Cumulative(pieces, solo).between(starts, starts + step_s) / step_s
    hidden = (
        Cumulative(pieces, ~pieces.all_known).between(starts, starts + step_s)
        > EPSILON_S
    )
    return np.where(hidden, np.nan, fraction)


def ego_rms_db(
    samples: np.ndarray,
    sample_rate: int,
    mask: np.ndarray,
    fraction: np.ndarray,
    cells: int,
    step_s: float,
    min_fraction: float,
) -> np.ndarray:
    """dBFS RMS of the masked samples of each cell; NaN below ``min_fraction``."""
    per_cell = round(sample_rate * step_s)
    result = np.full(cells, np.nan)
    available = min(cells, len(samples) // per_cell, len(mask) // per_cell)
    if not available:
        return result
    values = (
        samples[: available * per_cell].reshape(available, per_cell).astype(np.float64)
    )
    keep = mask[: available * per_cell].reshape(available, per_cell)
    count = keep.sum(axis=1)
    energy = (values**2 * keep).sum(axis=1) / np.maximum(count, 1)
    rms = np.sqrt(energy)
    db = np.where(rms > 0, 20.0 * np.log10(np.maximum(rms, 1e-12)), SILENCE_DB)
    ok = (count > 0) & (fraction[:available] >= min_fraction - 1e-9)
    result[:available] = np.where(ok, db, np.nan)
    return result


def pitch_cells(
    times: np.ndarray,
    f0: np.ndarray,
    mask: np.ndarray,
    fraction: np.ndarray,
    cell_starts: np.ndarray,
    step_s: float,
    min_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median voiced F0 and voiced fraction of the masked pitch frames of each cell."""
    cells = len(cell_starts)
    median = np.full(cells, np.nan)
    voiced_fraction = np.full(cells, np.nan)
    row = (
        np.floor((times - cell_starts[0]) / step_s + 1e-9).astype(np.int64)
        if cells
        else times
    )
    for cell in range(cells):
        if not fraction[cell] >= min_fraction - 1e-9:
            continue
        chosen = (row == cell) & mask
        if not chosen.any():
            continue
        values = f0[chosen]
        voiced = np.isfinite(values) & (values > 0)
        voiced_fraction[cell] = float(voiced.mean())
        if voiced.any():
            median[cell] = float(np.median(values[voiced]))
    return median, voiced_fraction


def frame_features(frame: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Per-frame controls: mean RGB (3), brightness, contrast, saturation, blur,
    dominant colour (3), difference to the previous frame (NaN for the first)."""
    rgb = frame.astype(np.float32)
    mean = rgb.mean(axis=(0, 1))
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    maximum, minimum = rgb.max(axis=-1), rgb.min(axis=-1)
    saturation = np.divide(
        maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0
    ).mean()
    laplacian = (
        -4 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    levels = np.minimum(
        (frame.astype(np.int32) * DOMINANT_LEVELS) // 256, DOMINANT_LEVELS - 1
    )
    codes = (
        levels[..., 0] * DOMINANT_LEVELS + levels[..., 1]
    ) * DOMINANT_LEVELS + levels[..., 2]
    top = int(np.bincount(codes.reshape(-1), minlength=DOMINANT_LEVELS**3).argmax())
    centre = (
        np.asarray(
            [
                top // DOMINANT_LEVELS**2,
                (top // DOMINANT_LEVELS) % DOMINANT_LEVELS,
                top % DOMINANT_LEVELS,
            ]
        )
        + 0.5
    ) * (256.0 / DOMINANT_LEVELS)
    difference = (
        float(np.abs(rgb - previous.astype(np.float32)).mean())
        if previous is not None
        else np.nan
    )
    return np.asarray(
        [
            *mean,
            luma.mean(),
            rgb.std(),
            saturation,
            laplacian.var(),
            *centre,
            difference,
        ],
        dtype=np.float64,
    )


FRAME_FEATURES = 11
"""Length of :func:`frame_features`' vector."""


def video_cell_features(
    per_frame: np.ndarray, cells: int, subframes: int
) -> dict[str, np.ndarray]:
    """Mean of the per-frame controls over the ``subframes`` frames of each cell."""
    available = min(cells, len(per_frame) // subframes)
    values = np.full((cells, FRAME_FEATURES), np.nan)
    if available:
        grouped = per_frame[: available * subframes].reshape(
            available, subframes, FRAME_FEATURES
        )
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # all-NaN difference of frame 0
            values[:available] = np.nanmean(grouped, axis=1)
    valid = np.zeros(cells, dtype=bool)
    valid[:available] = True
    return {
        "frame_mean_rgb": values[:, 0:3],
        "brightness": values[:, 3],
        "contrast": values[:, 4],
        "saturation": values[:, 5],
        "blur_score": values[:, 6],
        "dominant_colour": values[:, 7:10],
        "frame_difference": values[:, 10],
        "frame_valid": valid,
    }


# -- extractor assembly ------------------------------------------------------


@dataclass
class MediaResult:
    """One media extractor's tables and what its manifest must record."""

    tables: dict[Table, pa.Table]
    statistics: dict[str, Any]
    inputs: dict[str, Any]
    tools: dict[str, Any]
    materialized: set[str]
    unavailable: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MediaFile:
    """The file holding one recording's stream and its offset."""

    path: Path | None
    media_offset_s: float


def media_files(
    cfg: DictConfig, dataset: str, grid_sha256: str, stream: str
) -> tuple[dict[str, MediaFile], dict[str, Any]]:
    """Resolve each recording's audio or video file through the checked manifest."""
    from conv_wm.data.pipeline.media_manifest import media_manifest_for_grid

    manifest = media_manifest_for_grid(cfg, dataset, grid_sha256)
    if manifest is None:
        raise MissingPrerequisiteError(
            pipeline_paths(cfg).model_ready / dataset / "media_manifest.parquet",
            produce_with=f"conv-wm build media-manifest --dataset {dataset}",
        )
    root = pipeline_paths(cfg).raw / str(manifest.report["corpus_directory"])
    files: dict[str, MediaFile] = {}
    for row in manifest.table.to_dict(orient="records"):
        relative: str | None
        if stream == "audio":
            relative = (
                row["audio_path"]
                if pd.notna(row["audio_path"])
                else (
                    row["video_path"]
                    if bool(row.get("video_has_audio")) and pd.notna(row["video_path"])
                    else None
                )
            )
        else:
            relative = row["video_path"] if pd.notna(row["video_path"]) else None
        files[str(row["recording_id"])] = MediaFile(
            root / relative if relative else None, float(row["media_offset_s"])
        )
    inputs = {
        "media_manifest": {
            "file": manifest.table_path.name,
            "sha256": manifest.table_sha256,
        }
    }
    return files, inputs


def media_tables(
    cfg: DictConfig,
    dataset: str,
    extractor: Extractor,
    structures: Mapping[str, RecordingStructure],
    grid_keys: Mapping[str, GridKeys],
    config: LabelConfig,
    *,
    grid_sha256: str,
    external: bool,
    decode: Callable[[Path, float, float, int], np.ndarray] | None = None,
    frames: Callable[[Path, float, float, float, int], Iterator[np.ndarray]]
    | None = None,
) -> MediaResult:
    """Build the audio or video extractor of one dataset."""
    if decode is None and frames is None:
        require_ffmpeg()
    stream = "audio" if extractor is Extractor.AUDIO else "video"
    files, inputs = media_files(cfg, dataset, grid_sha256, stream)
    if extractor is Extractor.AUDIO:
        return _audio(
            structures,
            grid_keys,
            config,
            files,
            inputs,
            external,
            decode or decode_audio,
        )
    return _video(grid_keys, config, files, inputs, frames or iter_video_frames)


def _audio(
    structures: Mapping[str, RecordingStructure],
    grid_keys: Mapping[str, GridKeys],
    config: LabelConfig,
    files: Mapping[str, MediaFile],
    inputs: dict[str, Any],
    external: bool,
    decode: Callable[[Path, float, float, int], np.ndarray],
) -> MediaResult:
    rate = config.audio_sample_rate_hz
    parselmouth = (
        require_module("parselmouth", label="prosody.ego_f0_hz", extra="labels-audio")
        if external
        else None
    )
    step = DECISION_STEP_S
    parts: list[pa.Table] = []
    missing: list[str] = []
    for recording_id in sorted(structures):
        keys = grid_keys[recording_id]
        cells = len(keys.decision_index)
        starts = keys.decision_index * step
        media = files.get(recording_id)
        samples = np.empty(0, dtype=np.float32)
        if (
            media is not None
            and media.path is not None
            and media.path.exists()
            and cells
        ):
            start = float(starts[0]) + media.media_offset_s
            samples = decode(media.path, start, cells * step, rate)
            if (
                start < 0
            ):  # the media starts after the grid: pad, those cells are invalid
                samples = np.concatenate(
                    [np.full(round(-start * rate), np.nan, np.float32), samples]
                )
        else:
            missing.append(recording_id)
        finite = np.nan_to_num(samples, nan=0.0)
        features = audio_cell_features(finite, rate, cells, step)
        if len(samples):
            per_cell = round(rate * step)
            nan_cells = (
                np.isnan(samples[: (len(samples) // per_cell) * per_cell])
                .reshape(-1, per_cell)
                .any(axis=1)
            )
            features["audio_valid"][: len(nan_cells)] &= ~nan_cells[:cells]
        valid = features["audio_valid"]
        structure = structures[recording_id]
        fraction = ego_solo_fraction(structure, starts, step)
        sample_times = (
            starts[0] + np.arange(len(finite)) / rate if cells else np.empty(0)
        )
        mask = ego_solo_mask(structure, sample_times) & np.isfinite(samples)
        rms_db = ego_rms_db(
            finite, rate, mask, fraction, cells, step, config.audio_min_mask_fraction
        )
        columns: dict[str, pa.Array] = {}
        for name in (
            "global_audio_rms",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_flux",
            "background_noise_proxy",
        ):
            values = np.where(valid, features[name], np.nan)
            columns[name] = scalar(
                values.astype(np.float32), ~np.isfinite(values), pa.float32()
            )
        columns["spectral_valid"] = pa.array(features["spectral_valid"] & valid)
        columns["audio_valid"] = pa.array(valid)
        columns["ego_solo_mask_fraction"] = scalar(
            np.nan_to_num(fraction).astype(np.float32),
            ~np.isfinite(fraction),
            pa.float32(),
        )
        rms_db = np.where(valid, rms_db, np.nan)
        columns["ego_rms_db"] = scalar(
            np.nan_to_num(rms_db).astype(np.float32), ~np.isfinite(rms_db), pa.float32()
        )
        if parselmouth is not None:
            f0, voiced = _praat(
                parselmouth, finite, rate, structure, starts, fraction, config
            )
            f0 = np.where(valid, f0, np.nan)
            voiced = np.where(valid, voiced, np.nan)
            columns["ego_f0_hz"] = scalar(
                np.nan_to_num(f0).astype(np.float32), ~np.isfinite(f0), pa.float32()
            )
            columns["ego_voiced_fraction"] = scalar(
                np.nan_to_num(voiced).astype(np.float32),
                ~np.isfinite(voiced),
                pa.float32(),
            )
        parts.append(pa.table({**key_columns(recording_id, keys), **columns}))
    materialized = {
        "nuisance.global_audio_rms",
        "nuisance.zero_crossing_rate",
        "nuisance.spectral_centroid",
        "nuisance.spectral_bandwidth",
        "nuisance.spectral_flux",
        "nuisance.spectral_valid",
        "nuisance.background_noise_proxy",
        "nuisance.audio_valid",
        "prosody.ego_rms_db",
        "prosody.ego_solo_mask_fraction",
    }
    tools: dict[str, Any] = {}
    if parselmouth is not None:
        materialized |= {"prosody.ego_f0_hz", "prosody.ego_voiced_fraction"}
        tools["praat"] = {
            "tool": PRAAT.tool,
            "package": PRAAT.package,
            "package_version": importlib.metadata.version("praat-parselmouth"),
            "praat_version": getattr(parselmouth, "PRAAT_VERSION", None),
            "config": dict(PRAAT.config),
            "device": "cpu",
            "license": PRAAT.license,
            "determinism": PRAAT.determinism,
        }
    tools["ffmpeg"] = {"decoder": "ffmpeg", "sample_rate_hz": rate, "channels": 1}
    return MediaResult(
        tables={Table.GRID: pa.concat_tables(parts)} if parts else {},
        statistics={
            "recordings": len(parts),
            "recordings_without_media_file": len(missing),
            "recordings_without_media_file_examples": missing[:10],
        },
        inputs=inputs,
        tools=tools,
        materialized=materialized,
    )


def _praat(
    parselmouth: Any,
    samples: np.ndarray,
    rate: int,
    structure: RecordingStructure,
    starts: np.ndarray,
    fraction: np.ndarray,
    config: LabelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    settings = PRAAT.config
    if len(samples) < rate * 0.1:
        return np.full(len(starts), np.nan), np.full(len(starts), np.nan)
    sound = parselmouth.Sound(
        samples.astype(np.float64), sampling_frequency=float(rate)
    )
    pitch = sound.to_pitch_ac(
        time_step=settings["time_step_s"],
        pitch_floor=settings["pitch_floor_hz"],
        pitch_ceiling=settings["pitch_ceiling_hz"],
    )
    times = starts[0] + np.asarray(pitch.xs())
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
    f0[f0 == 0] = np.nan
    mask = ego_solo_mask(structure, times)
    return pitch_cells(
        times,
        f0,
        mask,
        fraction,
        starts,
        DECISION_STEP_S,
        config.audio_min_mask_fraction,
    )


def _video(
    grid_keys: Mapping[str, GridKeys],
    config: LabelConfig,
    files: Mapping[str, MediaFile],
    inputs: dict[str, Any],
    frames: Callable[[Path, float, float, float, int], Iterator[np.ndarray]],
) -> MediaResult:
    step = DECISION_STEP_S
    subframes = config.subframes_per_step
    rate = subframes / step
    parts: list[pa.Table] = []
    missing: list[str] = []
    for recording_id in sorted(grid_keys):
        keys = grid_keys[recording_id]
        cells = len(keys.decision_index)
        media = files.get(recording_id)
        vectors: list[np.ndarray] = []
        if (
            media is not None
            and media.path is not None
            and media.path.exists()
            and cells
        ):
            start = float(keys.decision_index[0]) * step + media.media_offset_s
            previous = None
            for image in frames(
                media.path, start, cells * step, rate, config.video_frame_width
            ):
                vectors.append(frame_features(image, previous))
                previous = image
                if len(vectors) >= cells * subframes:
                    break
        else:
            missing.append(recording_id)
        per_frame = np.asarray(vectors).reshape(-1, FRAME_FEATURES)
        features = video_cell_features(per_frame, cells, subframes)
        valid = features["frame_valid"]
        columns: dict[str, pa.Array] = {}
        for name in ("frame_mean_rgb", "dominant_colour"):
            values = features[name]
            columns[name] = pa.FixedSizeListArray.from_arrays(
                pa.array(
                    np.nan_to_num(values).astype(np.float32).reshape(-1), pa.float32()
                ),
                3,
                mask=pa.array(~valid),
            )
        for name in (
            "brightness",
            "contrast",
            "saturation",
            "blur_score",
            "frame_difference",
        ):
            values = np.where(valid, features[name], np.nan)
            columns[name] = scalar(
                np.nan_to_num(values).astype(np.float32),
                ~np.isfinite(values),
                pa.float32(),
            )
        columns["frame_valid"] = pa.array(valid)
        parts.append(pa.table({**key_columns(recording_id, keys), **columns}))
    return MediaResult(
        tables={Table.GRID: pa.concat_tables(parts)} if parts else {},
        statistics={
            "recordings": len(parts),
            "recordings_without_media_file": len(missing),
            "recordings_without_media_file_examples": missing[:10],
        },
        inputs=inputs,
        tools={
            "ffmpeg": {
                "decoder": "ffmpeg",
                "frame_rate_hz": rate,
                "frame_width": config.video_frame_width,
                "pixel_format": "rgb24",
            }
        },
        materialized={
            "nuisance.frame_mean_rgb",
            "nuisance.brightness",
            "nuisance.contrast",
            "nuisance.saturation",
            "nuisance.blur_score",
            "nuisance.dominant_colour",
            "nuisance.frame_difference",
            "nuisance.frame_valid",
        },
    )


__all__ = [
    "FRAME_FEATURES",
    "MediaResult",
    "MissingOptionalDependencyError",
    "audio_cell_features",
    "decode_audio",
    "ego_rms_db",
    "ego_solo_fraction",
    "ego_solo_mask",
    "frame_features",
    "iter_video_frames",
    "media_tables",
    "pitch_cells",
    "require_module",
    "video_cell_features",
]
