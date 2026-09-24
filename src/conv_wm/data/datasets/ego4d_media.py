"""Ego4D AV canonical recordings (clips) -> the full videos they were cut from.

The canonical ``recording_id`` is the AV ``clip_uid``; the release ships no
per-clip media, and the media metadata audit probes the source videos, named
by ``video_uid``. ``clips_clean`` maps one to the other and places the clip on
its video by frame index (``clip_media_offset_s``) — the same mapping the
native focal voice-state build uses to measure media coverage. Several clips
may share one video; each keeps its own offset.

The release's ``video_start_sec`` is not used: it is expressed on the
full-scale video's timeline (whose first frame can carry a non-zero PTS, e.g.
0.032 s) and sits one frame before ``video_start_frame`` for almost every
clip. Read against the downscaled ``video_540ss`` files, whose frame ``k`` is
at ``k / 30`` s, it misplaces the clip by up to ~0.1 s and can even fall
before the start of the file (-0.0013 s).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.media.media_manifest import (
    MediaFileIndex,
    MediaRecord,
    MediaSource,
)
from conv_wm.data.vocal.native_source import require_table

EGO4D_FPS = 30.0
"""Frame rate of every Ego4D video and of the AV clip timeline (frame / 30)."""


def clip_media_offset_s(clip: Mapping[Any, Any]) -> float:
    """Where clip time 0 sits on its source video file: ``media = clip + offset``.

    Clip frame ``f`` is video frame ``video_start_frame - clip_start_frame + f``,
    and both the clip timeline and the constant-rate source video place frame
    ``k`` at ``k / 30`` s.
    """
    frames = int(clip["video_start_frame"]) - int(clip["clip_start_frame"])
    return frames / EGO4D_FPS


def load_ego4d_media(
    cfg: DictConfig, media: pd.DataFrame, recording_ids: Sequence[str]
) -> MediaSource:
    """Resolve every requested clip to its source video and clip-to-video offset."""
    clips_path = get_path(cfg, "ego4d", "interim", "clips_clean")
    clips = {
        str(row["clip_uid"]): row
        for row in require_table(clips_path).to_dict(orient="records")
    }
    files = MediaFileIndex(media, "ego4d")
    records: list[MediaRecord] = []
    unresolved: dict[str, str] = {}
    for recording_id in sorted(set(recording_ids)):
        clip = clips.get(recording_id)
        if clip is None:
            unresolved[recording_id] = "clip_unknown"
            continue
        video, audio = files.resolve(str(clip["video_uid"]))
        if video is None and audio is None:
            unresolved[recording_id] = "media_missing"
            continue
        records.append(
            MediaRecord(
                recording_id, video, audio, media_offset_s=clip_media_offset_s(clip)
            )
        )
    return MediaSource(
        dataset="ego4d",
        records=tuple(records),
        unresolved=unresolved,
        input_paths=(clips_path,),
        mapping=(
            "recording_id is the AV clip_uid; its media is the source video named "
            "by clips_clean.video_uid, and media_offset_s = (video_start_frame - "
            "clip_start_frame) / 30 places the clip timeline on the video by frame "
            "index"
        ),
    )


__all__ = ["EGO4D_FPS", "clip_media_offset_s", "load_ego4d_media"]
