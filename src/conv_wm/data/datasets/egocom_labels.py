"""EgoCom speaker-attributed word timings -> canonical label facts.

One recording per point-of-view video, on the conversation part's clock (the
clock of the native focal voice state). Its participants are every speaker
of the part's transcript plus every speaker with a point-of-view video of the
part, identified by their EgoCom speaker id; the wearer is the video's
speaker. Each participant's speech is the union of their timed transcript
tokens — the same rule the native focal voice state applies to the wearer,
so the wearer's activity here is identical to it. Absence of a timed token is
SILENT; time the video's audio does not cover is UNKNOWN for everyone.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.datasets.egocom_cleaning import RULE_VERSION
from conv_wm.data.datasets.egocom_native_voice import (
    ANNOTATION_SCHEMA_VERSION,
    media_unknown,
)
from conv_wm.data.labels import facts
from conv_wm.data.labels.facts import (
    LabelFactsSpec,
    LabelSource,
    ParticipantFacts,
    RecordingFacts,
)
from conv_wm.data.vocal.native_source import (
    media_coverage_by_stem,
    require_table,
    timed_intervals,
)

PROVIDES = frozenset(
    {
        facts.SPEECH,
        facts.WORDS,
        facts.TRANSCRIPT,
        facts.MEDIA_AUDIO,
        facts.MEDIA_VIDEO,
        facts.META_BACKGROUND,
        facts.META_NATIVE_SPEAKER,
        facts.META_HOST,
    }
)


def _flag(value: object) -> bool | None:
    if value is None or pd.isna(value):  # type: ignore[arg-type]
        return None
    return bool(value)


def _speaker(value: object) -> str:
    return str(int(value))  # type: ignore[call-overload]


def load_egocom_label_facts(cfg: DictConfig, media: pd.DataFrame) -> LabelSource:
    """Every EgoCom point-of-view recording with all of its part's speakers."""
    info_path = get_path(cfg, "egocom", "interim", "video_info_clean")
    transcript_path = get_path(cfg, "egocom", "interim", "ground_truth_clean")
    info = require_table(info_path)
    transcript = require_table(transcript_path)
    coverage = media_coverage_by_stem(media, "egocom")
    speaker = pd.to_numeric(transcript["speaker_id"], errors="coerce")
    starts = pd.to_numeric(transcript["startTime"], errors="coerce")
    ends = pd.to_numeric(transcript["endTime"], errors="coerce")
    timed = starts.notna() & ends.notna() & (ends > starts)
    by_part = {
        str(part): group
        for part, group in transcript.groupby("conversation_id", sort=False)
    }
    views_by_part = {
        str(part): group for part, group in info.groupby("conversation_id", sort=False)
    }
    recordings: list[RecordingFacts] = []
    token_frames: list[pd.DataFrame] = []
    participants_without_view = 0
    for row in info.sort_values("video_name").to_dict(orient="records"):
        view_id = str(row["video_name"])
        part = str(row["conversation_id"])
        wearer = _speaker(row["video_speaker_id"])
        end_s = float(row["duration_seconds"])
        words = by_part.get(part, transcript.iloc[0:0])
        views = views_by_part[part]
        view_meta = {
            _speaker(item["video_speaker_id"]): item
            for item in views.to_dict(orient="records")
        }
        speakers = {_speaker(v) for v in speaker.reindex(words.index).dropna()}
        ids = sorted(speakers | set(view_meta), key=lambda s: (len(s), s))
        participants_without_view += len(set(ids) - set(view_meta))
        participants = []
        for pid in ids:
            own = words.loc[speaker.reindex(words.index).eq(int(pid))]
            meta = view_meta.get(pid, {})
            participants.append(
                ParticipantFacts(
                    participant_id=pid,
                    is_wearer=pid == wearer,
                    speech=timed_intervals(
                        own, "ground_truth_clean", "startTime", "endTime"
                    ),
                    metadata={
                        "native_speaker": _flag(meta.get("native_speaker")),
                        "is_host": _flag(meta.get("speaker_is_host")),
                    },
                )
            )
        recordings.append(
            RecordingFacts(
                dataset="egocom",
                recording_id=view_id,
                conversation_id=part,
                view_id=view_id,
                wearer_id=wearer,
                start_s=0.0,
                end_s=end_s,
                participants=tuple(participants),
                unknown=tuple(media_unknown(coverage.get(view_id), end_s)),
                metadata={
                    "background_fan": _flag(row.get("background_fan")),
                    "background_music": _flag(row.get("background_music")),
                },
            )
        )
        token_frames.append(
            pd.DataFrame(
                {
                    "recording_id": view_id,
                    "participant_id": speaker.reindex(words.index).map(
                        lambda v: _speaker(v) if pd.notna(v) else None
                    ),
                    "unit": "word",
                    "text": words["word"],
                    "start_s": starts.reindex(words.index).where(
                        timed.reindex(words.index)
                    ),
                    "end_s": ends.reindex(words.index).where(
                        timed.reindex(words.index)
                    ),
                    "timing_valid": timed.reindex(words.index).to_numpy(bool),
                    "source_row": words["source_row"].astype("int64"),
                }
            )
        )
    tokens = (
        pd.concat(token_frames, ignore_index=True)
        if token_frames
        else pd.DataFrame(columns=list(facts.TOKEN_COLUMNS))
    )
    return LabelSource(
        dataset="egocom",
        recordings=recordings,
        annotation_paths=(info_path, transcript_path),
        annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
        cleaning_rule_version=RULE_VERSION,
        tokens=tokens,
        statistics={
            "recordings": len(recordings),
            "participant_rows_without_view": participants_without_view,
            "untimed_token_rows": int((~timed).sum()),
        },
        limitations=(
            (
                "Every participant's SPEAKING is the union of their timed transcript "
                "tokens; untimed tokens (punctuation, empty and some words) carry no "
                "time and never create speech, so absence of annotation is SILENT."
            ),
            (
                "Seven two-speaker conversations attribute speech to speaker 3, who "
                "has no recording; that speaker is kept as a participant with their "
                "own id (never merged into another participant), its status being "
                "unresolved in the corpus."
            ),
        ),
    )


EGOCOM_LABEL_FACTS = LabelFactsSpec(provides=PROVIDES, load=load_egocom_label_facts)

__all__ = ["EGOCOM_LABEL_FACTS", "PROVIDES", "load_egocom_label_facts"]
