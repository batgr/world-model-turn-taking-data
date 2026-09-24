"""EgoCom canonical recordings -> their point-of-view video files.

The canonical ``recording_id`` is ``video_info.video_name`` (see
:mod:`conv_wm.data.datasets.egocom_native_voice`), which is the file stem of
the POV video the media metadata audit probed — the same key the native
focal voice-state build uses to find a recording's media. The recording's
canonical clock is the video's own, so the media offset is zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from omegaconf import DictConfig

from conv_wm.data.media.media_manifest import (
    MediaFileIndex,
    MediaRecord,
    MediaSource,
)


def load_egocom_media(
    cfg: DictConfig, media: pd.DataFrame, recording_ids: Sequence[str]
) -> MediaSource:
    """Resolve every requested EgoCom recording to the files whose stem is its id."""
    del cfg  # the recording id is the media key; no annotation table is needed
    files = MediaFileIndex(media, "egocom")
    records: list[MediaRecord] = []
    unresolved: dict[str, str] = {}
    for recording_id in sorted(set(recording_ids)):
        video, audio = files.resolve(recording_id)
        if video is None and audio is None:
            unresolved[recording_id] = "media_missing"
            continue
        records.append(MediaRecord(recording_id, video, audio, media_offset_s=0.0))
    return MediaSource(
        dataset="egocom",
        records=tuple(records),
        unresolved=unresolved,
        mapping=(
            "recording_id is the video_info video_name, the stem of the POV video; "
            "the canonical clock is the video's own timeline (offset 0)"
        ),
    )


__all__ = ["load_egocom_media"]
