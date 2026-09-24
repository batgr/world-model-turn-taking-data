"""Native transcripts -> token rows and the cues derivable from them without a model.

Tokens are kept exactly as transcribed, untimed ones included. Only cues that
follow deterministically from the transcription are derived: speech rate
(word timings only), the closing punctuation mark, whether it is a question
mark, the closing lexical token, and discourse markers from a fixed lexicon.
Completion, dialogue acts, repair and the like need a validated model or a
human annotation and stay unsupported in the registry.

A *text unit* is a turn of the speech extractor when the corpus times words
(tokens are placed in the turn of their speaker containing their midpoint),
or a native utterance when it only times utterances.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from conv_wm.data.labels.timeline import EPSILON_S, RecordingStructure

TEXT_RULES_VERSION = 1
"""Bumped when the token classes, the association or the lexicon change."""

DISCOURSE_MARKERS_INITIAL = (
    "you know",
    "i mean",
    "so",
    "well",
    "okay",
    "ok",
    "oh",
    "right",
    "anyway",
    "actually",
    "like",
    "and",
    "but",
)
DISCOURSE_MARKERS_FINAL = (
    "you know",
    "i guess",
    "i think",
    "or something",
    "right",
    "okay",
    "so",
    "though",
)
"""Fixed lexicons (``TEXT_RULES_VERSION`` 1); multi-word entries are tried first."""

FINAL_PUNCTUATION = (".", "?", "!", ",")
_LETTER = re.compile(r"[A-Za-z]")
_PUNCTUATION = re.compile(r"^[^\w\s]+$")
_SPECIAL = re.compile(r"^(?:\[[^]]*\]|<[^>]*>|\([^)]*\))$")
_STRIP = re.compile(r"^[^\w']+|[^\w']+$")

TOKEN_EVENT_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("event_type", pa.string()),
        ("time_s", pa.float64()),
        ("participant_index", pa.int16()),
        ("participant_id", pa.string()),
        ("unit", pa.string()),
        ("text", pa.string()),
        ("token_class", pa.string()),
        ("end_s", pa.float64()),
        ("timing_valid", pa.bool_()),
        ("turn_id", pa.int64()),
        ("source_row", pa.int64()),
    ]
)

TEXT_UNIT_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("segment_type", pa.string()),
        ("segment_id", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("duration_s", pa.float64()),
        ("participant_index", pa.int16()),
        ("participant_id", pa.string()),
        ("unit", pa.string()),
        ("word_count", pa.int32()),
        ("timed_word_count", pa.int32()),
        ("words_per_second", pa.float64()),
        ("words_per_voiced_second", pa.float64()),
        ("speech_rate_valid", pa.bool_()),
        ("final_punctuation", pa.string()),
        ("interrogative_cue", pa.bool_()),
        ("final_token", pa.string()),
        ("initial_discourse_marker", pa.string()),
        ("final_discourse_marker", pa.string()),
    ]
)


def token_text(value: Any) -> str | None:
    """A token's text, ``None`` for a missing (null / NaN) transcription."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def token_class(text: str | None) -> str:
    """``lexical``, ``punctuation``, ``special`` (bracketed noise tags) or ``empty``."""
    value = (token_text(text) or "").strip()
    if not value:
        return "empty"
    if _SPECIAL.match(value):
        return "special"
    if _LETTER.search(value):
        return "lexical"
    if _PUNCTUATION.match(value):
        return "punctuation"
    return "other"


def normalize(word: str) -> str:
    """Lower-case a lexical token and strip surrounding punctuation."""
    return _STRIP.sub("", word.strip().lower())


def discourse_marker(
    words: list[str], lexicon: tuple[str, ...], *, initial: bool
) -> str | None:
    """The first lexicon entry matching the opening (or closing) words."""
    for entry in sorted(lexicon, key=lambda item: -len(item.split())):
        size = len(entry.split())
        if len(words) < size:
            continue
        window = words[:size] if initial else words[-size:]
        if " ".join(window) == entry:
            return entry
    return None


def closing_punctuation(text: str) -> str | None:
    """The last punctuation mark of a unit, if it closes it."""
    stripped = text.rstrip()
    if stripped and stripped[-1] in FINAL_PUNCTUATION:
        return stripped[-1]
    return None


def text_rows(
    structure: RecordingStructure, tokens: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Token event rows and text-unit segment rows of one recording."""
    ids = structure.pieces.participant_ids
    index = {pid: i for i, pid in enumerate(ids)}
    recording_id = structure.recording.recording_id
    frame = tokens.sort_values(["source_row"], kind="stable").reset_index(drop=True)
    timed = frame["timing_valid"].astype(bool).to_numpy()
    starts = pd.to_numeric(frame["start_s"], errors="coerce").to_numpy(float)
    ends = pd.to_numeric(frame["end_s"], errors="coerce").to_numpy(float)
    turn_ids = np.full(len(frame), -1, dtype=np.int64)
    turns = structure.turns
    for ordinal, turn in enumerate(turns):
        pid = ids[turn.participant]
        own = frame["participant_id"].eq(pid).to_numpy() & timed
        if frame["unit"].eq("word").all():
            middle = (starts + ends) / 2.0
            inside = (
                own
                & (middle >= turn.start_s - EPSILON_S)
                & (middle < turn.end_s + EPSILON_S)
            )
            turn_ids[inside & (turn_ids < 0)] = ordinal
    if len(frame) and not frame["unit"].eq("word").all():
        for position in np.flatnonzero(timed):
            pid = _participant(frame.at[position, "participant_id"])
            best, best_overlap = -1, 0.0
            for ordinal, turn in enumerate(turns):
                if ids[turn.participant] != pid:
                    continue
                overlap = min(turn.end_s, ends[position]) - max(
                    turn.start_s, starts[position]
                )
                if overlap > best_overlap + EPSILON_S:
                    best, best_overlap = ordinal, overlap
            turn_ids[position] = best
    classes = [token_class(value) for value in frame["text"].tolist()]
    events: list[dict[str, Any]] = []
    for position, item in enumerate(frame.to_dict(orient="records")):
        pid = _participant(item["participant_id"])
        events.append(
            {
                "recording_id": recording_id,
                "event_type": "token",
                "time_s": float(starts[position]) if timed[position] else None,
                "participant_index": index.get(pid) if pid is not None else None,
                "participant_id": pid,
                "unit": str(item["unit"]),
                "text": token_text(item["text"]),
                "token_class": classes[position],
                "end_s": float(ends[position]) if timed[position] else None,
                "timing_valid": bool(timed[position]),
                "turn_id": int(turn_ids[position]) if turn_ids[position] >= 0 else None,
                "source_row": int(item["source_row"]),
            }
        )
    frame = frame.assign(token_class=classes, turn_id=turn_ids, start=starts, end=ends)
    units: list[dict[str, Any]] = []
    if len(frame) and frame["unit"].eq("word").all():
        groups = {
            int(str(key)): group.sort_values(["start", "source_row"], kind="stable")
            for key, group in frame[frame["turn_id"] >= 0].groupby(
                "turn_id", sort=False
            )
        }
        for ordinal, turn in enumerate(turns):
            own = groups.get(ordinal, frame.iloc[0:0])
            voiced = turn.duration_s - sum(turn.pauses)
            texts = [token_text(value) or "" for value in own["text"].tolist()]
            lexical = [
                normalize(t)
                for t, c in zip(texts, own["token_class"], strict=True)
                if c == "lexical"
            ]
            closing = texts[-1].strip() if texts else ""
            punctuation = closing if closing in FINAL_PUNCTUATION else None
            count = len(lexical)
            units.append(
                _unit(
                    recording_id,
                    ordinal,
                    turn.start_s,
                    turn.end_s,
                    turn.participant,
                    ids[turn.participant],
                    "turn",
                    lexical,
                    punctuation,
                    words_per_second=count / turn.duration_s
                    if count and turn.duration_s > 0
                    else None,
                    words_per_voiced_second=count / voiced
                    if count and voiced > 0
                    else None,
                )
            )
    elif len(frame):
        for ordinal, item in enumerate(
            frame[frame["unit"].eq("utterance") & frame["timing_valid"].astype(bool)]
            .sort_values(["start", "source_row"], kind="stable")
            .to_dict(orient="records")
        ):
            text = token_text(item["text"]) or ""
            lexical = [
                normalize(w) for w in text.split() if token_class(w) == "lexical"
            ]
            pid = _participant(item["participant_id"])
            units.append(
                _unit(
                    recording_id,
                    ordinal,
                    float(item["start"]),
                    float(item["end"]),
                    index.get(pid) if pid is not None else None,
                    pid,
                    "utterance",
                    lexical,
                    closing_punctuation(text),
                    words_per_second=None,
                    words_per_voiced_second=None,
                )
            )
    return events, units


def _participant(value: Any) -> str | None:
    """A token's participant id, ``None`` when the transcript leaves it unresolved."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def _unit(
    recording_id: str,
    ordinal: int,
    start_s: float,
    end_s: float,
    participant_index: int | None,
    participant_id: str | None,
    unit: str,
    lexical: list[str],
    punctuation: str | None,
    *,
    words_per_second: float | None,
    words_per_voiced_second: float | None,
) -> dict[str, Any]:
    lexical = [word for word in lexical if word]
    has_words = bool(lexical)
    return {
        "recording_id": recording_id,
        "segment_type": "text_unit",
        "segment_id": ordinal,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "participant_index": participant_index,
        "participant_id": participant_id,
        "unit": unit,
        "word_count": len(lexical),
        "timed_word_count": len(lexical) if unit == "turn" else None,
        "words_per_second": words_per_second,
        "words_per_voiced_second": words_per_voiced_second,
        "speech_rate_valid": unit == "turn" and words_per_second is not None,
        "final_punctuation": punctuation if has_words else None,
        "interrogative_cue": (punctuation == "?") if has_words else None,
        "final_token": lexical[-1] if has_words else None,
        "initial_discourse_marker": discourse_marker(
            lexical, DISCOURSE_MARKERS_INITIAL, initial=True
        )
        if has_words
        else None,
        "final_discourse_marker": discourse_marker(
            lexical, DISCOURSE_MARKERS_FINAL, initial=False
        )
        if has_words
        else None,
    }


__all__ = [
    "DISCOURSE_MARKERS_FINAL",
    "DISCOURSE_MARKERS_INITIAL",
    "TEXT_RULES_VERSION",
    "TEXT_UNIT_SCHEMA",
    "TOKEN_EVENT_SCHEMA",
    "closing_punctuation",
    "discourse_marker",
    "normalize",
    "text_rows",
    "token_class",
    "token_text",
]
