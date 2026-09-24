"""Ego4D AV v2 annotations -> canonical label facts.

One recording per clip with a camera wearer, on the clip timeline of the
native focal voice state. Participants are the clip's annotated persons
(``persons``), identified by their Ego4D person id; each speaks inside their
own ``voice_segments``, so the wearer's activity is identical to the native
focal voice state. An invalid clip and time the source video's audio does not
cover are UNKNOWN for everyone; an explicitly missing voice region is UNKNOWN
for the person it names, or for everyone when it names nobody.

Voice segments of a person absent from ``persons`` become that person's own
participant; a person id of ``-1`` becomes the unresolved pseudo-participant.
Social (LAM/TTM) segments and face tracks are passed through with their
native semantics; rows whose person does not resolve keep ``participant_id``
null.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.datasets.ego4d_cleaning import RULE_VERSION
from conv_wm.data.datasets.ego4d_media import EGO4D_FPS, clip_media_offset_s
from conv_wm.data.datasets.ego4d_native_voice import (
    ANNOTATION_SCHEMA_VERSION,
    media_unknown,
)
from conv_wm.data.labels import facts
from conv_wm.data.labels.facts import (
    UNRESOLVED_PARTICIPANT_ID,
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
from conv_wm.data.vocal.native_state import NativeAnnotation

PROVIDES = frozenset(
    {
        facts.SPEECH,
        facts.TRANSCRIPT,
        facts.SOCIAL_LOOKING,
        facts.SOCIAL_TALKING,
        facts.FACE_TRACKS,
        facts.MEDIA_AUDIO,
        facts.MEDIA_VIDEO,
        facts.META_SOURCE_OFFSET,
    }
)
UNRESOLVED_PERSONS = frozenset({"-1", ""})


def _person(value: object) -> str | None:
    if value is None or pd.isna(value):  # type: ignore[arg-type]
        return None
    text = str(value)
    return None if text in UNRESOLVED_PERSONS else text


def _groups(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(uid): group for uid, group in table.groupby("clip_uid", sort=False)}


def load_ego4d_label_facts(cfg: DictConfig, media: pd.DataFrame) -> LabelSource:
    """Every Ego4D clip with a camera wearer, with all of its annotated persons."""
    paths = {
        key: get_path(cfg, "ego4d", "interim", key)
        for key in (
            "clips_clean",
            "persons_clean",
            "voice_segments_clean",
            "missing_voice_segments_clean",
            "transcriptions_clean",
            "social_segments_talking_clean",
            "social_segments_looking_clean",
            "tracks_clean",
        )
    }
    tables = {key: require_table(path) for key, path in paths.items()}
    clips = tables["clips_clean"]
    persons = tables["persons_clean"]
    coverage = media_coverage_by_stem(media, "ego4d")
    persons_by_clip = _groups(persons)
    voice_by_clip = _groups(tables["voice_segments_clean"])
    missing_by_clip = _groups(tables["missing_voice_segments_clean"])
    empty_voice = tables["voice_segments_clean"].iloc[0:0]
    empty_missing = tables["missing_voice_segments_clean"].iloc[0:0]
    recordings: list[RecordingFacts] = []
    clip_participants: dict[str, set[str]] = {}
    statistics = {
        "clips_without_camera_wearer": 0,
        "voice_persons_absent_from_persons": 0,
        "clips_with_unresolved_speech": 0,
    }
    for row in clips.sort_values("clip_uid").to_dict(orient="records"):
        clip_uid = str(row["clip_uid"])
        clip_persons = persons_by_clip.get(clip_uid, persons.iloc[0:0])
        wearers = clip_persons.loc[
            clip_persons["is_camera_wearer"].astype(bool), "person_id"
        ]
        if wearers.empty:
            statistics["clips_without_camera_wearer"] += 1
            continue
        wearer = str(wearers.iloc[-1])  # the native focal adapter keeps the last one
        start_s, end_s = float(row["clip_start_sec"]), float(row["clip_end_sec"])
        voice = voice_by_clip.get(clip_uid, empty_voice)
        missing = missing_by_clip.get(clip_uid, empty_missing)
        voice_person = voice["person_id"].map(_person)
        missing_person = (
            missing["person_id"].map(_person) if len(missing) else missing["person_id"]
        )
        ids = {str(p) for p in clip_persons["person_id"]}
        extra = {p for p in voice_person.dropna() if p not in ids}
        statistics["voice_persons_absent_from_persons"] += len(extra)
        participants = [
            ParticipantFacts(
                participant_id=pid,
                is_wearer=pid == wearer,
                speech=timed_intervals(
                    voice.loc[voice_person.eq(pid)],
                    "voice_segments_clean",
                    "start_time",
                    "end_time",
                ),
                unknown=timed_intervals(
                    missing.loc[missing_person.eq(pid)] if len(missing) else missing,
                    "missing_voice_segments_clean",
                    "start_time",
                    "end_time",
                ),
            )
            for pid in sorted(ids | extra)
        ]
        unresolved = voice.loc[voice_person.isna()]
        if len(unresolved):
            statistics["clips_with_unresolved_speech"] += 1
            participants.append(
                ParticipantFacts(
                    participant_id=UNRESOLVED_PARTICIPANT_ID,
                    is_wearer=False,
                    speech=timed_intervals(
                        unresolved, "voice_segments_clean", "start_time", "end_time"
                    ),
                    resolved=False,
                )
            )
        unknown: list[NativeAnnotation] = []
        if not bool(row["valid"]):
            unknown.append(NativeAnnotation(start_s, end_s, "clip_invalid"))
        if len(missing):
            unknown.extend(
                timed_intervals(
                    missing.loc[missing_person.isna()],
                    "missing_voice_segments_clean",
                    "start_time",
                    "end_time",
                )
            )
        unknown.extend(
            media_unknown(
                coverage.get(str(row["video_uid"])),
                clip_start_s=start_s,
                clip_end_s=end_s,
                media_offset_s=clip_media_offset_s(row),
            )
        )
        clip_participants[clip_uid] = ids | extra
        recordings.append(
            RecordingFacts(
                dataset="ego4d",
                recording_id=clip_uid,
                conversation_id=clip_uid,
                view_id=str(row["video_uid"]),
                wearer_id=wearer,
                start_s=start_s,
                end_s=end_s,
                participants=tuple(participants),
                unknown=tuple(unknown),
                frame_rate_hz=EGO4D_FPS,
                metadata={"source_offset_s": float(row["video_start_sec"])},
            )
        )
    kept = set(clip_participants)
    return LabelSource(
        dataset="ego4d",
        recordings=recordings,
        annotation_paths=tuple(paths.values()),
        annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
        cleaning_rule_version=RULE_VERSION,
        tokens=_tokens(tables["transcriptions_clean"], kept),
        social=_social(
            tables["social_segments_talking_clean"],
            tables["social_segments_looking_clean"],
            clip_participants,
        ),
        tracks=_tracks(tables["tracks_clean"], clip_participants),
        statistics={"recordings": len(recordings), **statistics},
        limitations=(
            (
                "Every participant's SPEAKING is their native AV voice segments; a "
                "segment may absorb short internal pauses and is never refined "
                "acoustically. Speech the annotators did not mark is SILENT."
            ),
            (
                "Transcriptions are timed per utterance only: word-level labels "
                "(speech rate) are unsupported for Ego4D."
            ),
            (
                "Looking-At-Me is only defined on tracked faces and Talking-To-Me only "
                "inside native talking segments; the raw 'target' field is preserved "
                "and never interpreted as an addressee."
            ),
        ),
    )


def _tokens(table: pd.DataFrame, clips: set[str]) -> pd.DataFrame:
    rows = table.loc[table["clip_uid"].astype(str).isin(clips)]
    starts = pd.to_numeric(rows["start_time_sec"], errors="coerce")
    ends = pd.to_numeric(rows["end_time_sec"], errors="coerce")
    timed = starts.notna() & ends.notna() & (ends > starts)
    return pd.DataFrame(
        {
            "recording_id": rows["clip_uid"].astype(str),
            "participant_id": rows["person_id"].map(_person),
            "unit": "utterance",
            "text": rows["transcription"],
            "start_s": starts.where(timed),
            "end_s": ends.where(timed),
            "timing_valid": timed.to_numpy(bool),
            "source_row": rows.index.to_numpy(dtype="int64"),
        }
    ).reset_index(drop=True)


def _social(
    talking: pd.DataFrame, looking: pd.DataFrame, participants: dict[str, set[str]]
) -> pd.DataFrame:
    frames = []
    for kind, table in (("talking", talking), ("looking", looking)):
        rows = table.loc[table["clip_uid"].astype(str).isin(participants)]
        person = rows["person"].map(_person)
        resolved = [
            p if p is not None and p in participants[str(c)] else None
            for p, c in zip(person, rows["clip_uid"], strict=True)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "recording_id": rows["clip_uid"].astype(str),
                    "participant_id": resolved,
                    "person": rows["person"].astype(str),
                    "kind": kind,
                    "start_s": rows["start_time"].astype(float),
                    "end_s": rows["end_time"].astype(float),
                    "start_frame": rows["start_frame"].astype("int64"),
                    "end_frame": rows["end_frame"].astype("int64"),
                    "is_at_me": rows["is_at_me"],
                    "annotation_target": rows["target"],
                    "source_row": rows.index.to_numpy(dtype="int64"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _tracks(table: pd.DataFrame, participants: dict[str, set[str]]) -> pd.DataFrame:
    rows = table.loc[table["clip_uid"].astype(str).isin(participants)]
    return pd.DataFrame(
        {
            "recording_id": rows["clip_uid"].astype(str),
            "participant_id": rows["person_id"].astype(str),
            "track_id": rows["track_id"].astype(str),
            "frame": rows["clip_frame"].astype("int64"),
            "time_s": rows["clip_frame"].astype(float) / EGO4D_FPS,
            "x": rows["x"].astype(float),
            "y": rows["y"].astype(float),
            "width": rows["width"].astype(float),
            "height": rows["height"].astype(float),
        }
    ).reset_index(drop=True)


EGO4D_LABEL_FACTS = LabelFactsSpec(provides=PROVIDES, load=load_ego4d_label_facts)

__all__ = ["EGO4D_LABEL_FACTS", "PROVIDES", "load_ego4d_label_facts"]
