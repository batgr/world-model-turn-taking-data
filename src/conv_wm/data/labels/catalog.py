"""The label catalog: every label the data layer knows about.

This file is data. Semantics are stated once here and nowhere else: the
generated registry page (``docs/data_pipeline/labels_registry.md``), the
``registry.json`` shipped next to every label artifact, the coverage report
and the selection API all read :data:`REGISTRY`.

Conventions used by the descriptions (see ``docs/data_pipeline/labels.md``):

* ``Δ`` is the action-grid decision step (0.1 s) and ``S`` the configured
  ``subframes_per_step`` (default 3, i.e. 30 Hz subframes, one per video frame
  at 30 fps); ``H`` is the number of configured future horizons.
* ``participant_index`` indexes the recording's ``participant_ids``; index 0
  is always the camera wearer ("ego").
* Floor-holder codes: ``>= 0`` a participant index, ``-1`` NONE (nobody
  speaking), ``-2`` MULTIPLE (overlap). ``null`` is always UNKNOWN.
"""

from __future__ import annotations

from typing import Any

from conv_wm.data.labels import facts
from conv_wm.data.labels.registry import (
    ExternalTool,
    Extractor,
    LabelSpec,
    Level,
    Modality,
    Role,
    SourceKind,
    Table,
    validate,
)

A, T, V, M = Modality.AUDIO, Modality.TEXT, Modality.VIDEO, Modality.METADATA
NATIVE = SourceKind.NATIVE_ANNOTATION
DET = SourceKind.DETERMINISTIC
EXT = SourceKind.EXTERNAL_MODEL
HUMAN = SourceKind.HUMAN_ANNOTATION

SUBFRAME_REF = (
    "subframe j of cell k: [t_k + j·Δ/S, t_k + (j+1)·Δ/S), t_k = k·Δ on the "
    "recording's canonical clock"
)
CELL_REF = "decision cell k: [t_k, t_k + Δ), t_k = k·Δ"
CELL_END_REF = (
    "reference instant r_k = t_k + Δ (cell end): everything strictly before r_k "
    "is past; an event at exactly r_k is future"
)
HORIZON_REF = (
    "future window [r_k, r_k + h) for each configured horizon h, r_k = t_k + Δ"
)
NATIVE_POINT_REF = "native canonical timestamp in seconds, never rounded to the grid"
NATIVE_INTERVAL_REF = "native canonical interval [start_s, end_s) in seconds"
RECORDING_REF = "the whole recording (all of its observed time)"

BOOL_PS = "list<fixed_size_list<bool, S>>"
BOOL_S = "fixed_size_list<bool, S>"
INT8_S = "fixed_size_list<int8, S>"
INT16_S = "fixed_size_list<int16, S>"
FLOAT_S = "fixed_size_list<float32, S>"
PS = "[participant, subframe]"
SF = "[subframe]"

UNKNOWN_NULL = "null where the state of a required participant is UNKNOWN in the unit."
TRANSITION_NULL = (
    "null where the transition is not observable (participant UNKNOWN in the "
    "subframe or the preceding one, or the subframe opens the recording)."
)
AT_EGO_ONSET = (
    "null except in subframes holding a wearer onset; see "
    "onset_context.ego_onset_context_valid_subframes."
)

PRAAT = ExternalTool(
    tool="Praat pitch tracker (autocorrelation, via Parselmouth)",
    package="praat-parselmouth",
    extra="labels-audio",
    checkpoint=None,
    config={
        "method": "to_pitch_ac",
        "time_step_s": 0.01,
        "pitch_floor_hz": 75.0,
        "pitch_ceiling_hz": 500.0,
    },
    license=(
        "Parselmouth and Praat are GPL-3.0; the licence covers the software, not "
        "the numeric outputs, which may be redistributed with the derived data."
    ),
    reference=(
        "Jadoul, Thompson & de Boer (2018), Introducing Parselmouth, J. Phonetics; "
        "Boersma & Weenink, Praat. https://parselmouth.readthedocs.io"
    ),
    determinism="Deterministic signal processing: identical PCM gives identical output.",
)
OPENSMILE = ExternalTool(
    tool="openSMILE eGeMAPSv02 low-level descriptors",
    package="opensmile",
    extra=None,
    checkpoint="eGeMAPSv02",
    config={"feature_level": "LowLevelDescriptors"},
    license=(
        "audEERING openSMILE licence: free for private, research and educational "
        "use only; commercial use needs a separate licence. Check before any "
        "redistribution of derived features."
    ),
    reference=(
        "Eyben et al. (2016), The Geneva Minimalistic Acoustic Parameter Set "
        "(GeMAPS), IEEE TAC. https://audeering.github.io/opensmile-python/"
    ),
    determinism="Deterministic signal processing.",
)
MEDIAPIPE = ExternalTool(
    tool="MediaPipe Face Landmarker",
    package="mediapipe",
    extra="labels-video",
    checkpoint="face_landmarker.task (float16, user-supplied path, sha256 recorded)",
    config={"running_mode": "VIDEO", "num_faces": 4},
    license="Apache-2.0 (library and published model bundle).",
    reference="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
    determinism="Deterministic on CPU for a fixed model file; GPU delegates may differ.",
)
MMPOSE = ExternalTool(
    tool="MMPose whole-body (COCO-WholeBody) pose estimation",
    package="mmpose",
    extra="labels-video",
    checkpoint="a COCO-WholeBody checkpoint (user-supplied, sha256 recorded)",
    config={"keypoints": "COCO-WholeBody 133"},
    license="Apache-2.0 (library); checkpoint terms follow their training data.",
    reference="https://mmpose.readthedocs.io (whole-body demos)",
    determinism="Deterministic on CPU for a fixed checkpoint; GPU kernels may differ.",
)
WHISPERX = ExternalTool(
    tool="WhisperX ASR + word alignment",
    package="whisperx",
    extra="labels-text",
    checkpoint="a Whisper checkpoint and a wav2vec2 alignment model",
    config={"language": "en"},
    license="BSD-2-Clause (WhisperX); Whisper weights MIT; alignment models vary.",
    reference="Bain et al. (2023), WhisperX. https://github.com/m-bain/whisperX",
    determinism="Beam search on fixed weights is deterministic on CPU; GPU may differ.",
)
PYANNOTE = ExternalTool(
    tool="pyannote.audio speaker diarization",
    package="pyannote.audio",
    extra="labels-audio",
    checkpoint="a pyannote diarization pipeline (gated on Hugging Face)",
    config={},
    license="MIT (library); pretrained pipelines carry gated user conditions.",
    reference="Bredin et al., pyannote.audio. https://github.com/pyannote/pyannote-audio",
    determinism="Deterministic on CPU for a fixed pipeline; clustering may vary on GPU.",
)
ANNOTATION_NEEDED = ExternalTool(
    tool="none validated",
    package=None,
    extra=None,
    checkpoint=None,
    config={},
    license="n/a",
    reference="n/a",
    determinism="n/a",
)


def _available(
    name: str,
    description: str,
    *,
    level: Level,
    modalities: tuple[Modality, ...],
    dtype: str,
    shape: str,
    time_reference: str,
    source_kind: SourceKind,
    role: Role,
    validity: str,
    extractor: Extractor,
    table: Table,
    requires: tuple[str, ...] = (facts.SPEECH,),
    columns: tuple[str, ...] | None = None,
    row_filter: tuple[str, str] | None = None,
    external: ExternalTool | None = None,
    notes: str = "",
) -> LabelSpec:
    return LabelSpec(
        name=name,
        family=name.split(".", 1)[0],
        description=description,
        level=level,
        modalities=modalities,
        dtype=dtype,
        shape=shape,
        time_reference=time_reference,
        source_kind=source_kind,
        role=role,
        validity=validity,
        requires=requires,
        extractor=extractor,
        table=table,
        columns=columns if columns is not None else (name.split(".", 1)[1],),
        row_filter=row_filter,
        external=external,
        notes=notes,
    )


def _unsupported(
    name: str,
    description: str,
    *,
    level: Level,
    modalities: tuple[Modality, ...],
    source_kind: SourceKind,
    reason: str,
    role: Role = Role.CONTEXT,
    external: ExternalTool | None = None,
    dtype: str = "n/a",
    shape: str = "n/a",
    time_reference: str = "n/a (not materialized)",
    notes: str = "",
) -> LabelSpec:
    if source_kind is EXT and external is None:
        external = ANNOTATION_NEEDED
    return LabelSpec(
        name=name,
        family=name.split(".", 1)[0],
        description=description,
        level=level,
        modalities=modalities,
        dtype=dtype,
        shape=shape,
        time_reference=time_reference,
        source_kind=source_kind,
        role=role,
        validity="not materialized: requesting it explicitly is an error",
        external=external,
        unsupported_reason=reason,
        notes=notes,
    )


def _speech_grid(
    name: str,
    description: str,
    *,
    dtype: str,
    shape: str,
    validity: str,
    source_kind: SourceKind = DET,
    role: Role = Role.CONTEXT,
    level: Level = Level.SUBFRAME,
    time_reference: str = SUBFRAME_REF,
    columns: tuple[str, ...] | None = None,
    notes: str = "",
) -> LabelSpec:
    return _available(
        name,
        description,
        level=level,
        modalities=(A,),
        dtype=dtype,
        shape=shape,
        time_reference=time_reference,
        source_kind=source_kind,
        role=role,
        validity=validity,
        extractor=Extractor.SPEECH,
        table=Table.GRID,
        columns=columns,
        notes=notes,
    )


def _timing(name: str, description: str, *, target: bool) -> LabelSpec:
    column = name.split(".", 1)[1]
    since = column.startswith("time_since")
    return _speech_grid(
        name,
        description,
        dtype="float32 seconds (+ bool valid)",
        shape="[]",
        level=Level.GRID,
        time_reference=CELL_END_REF,
        role=Role.TARGET if target else Role.CONTEXT,
        columns=(column, f"{column}_valid"),
        validity=(
            (
                "null and valid=false when no such event occurred since the start of "
                "the current observed span (left-censored: the true value is at least "
                "timing.observation_bounds.time_since_observation_start) or when r_k "
                "is not observed. Never zero-filled."
            )
            if since
            else (
                "null and valid=false when no such event occurs before the end of the "
                "current observed span (right-censored at recording end or UNKNOWN: "
                "the true value is at least "
                "timing.observation_bounds.time_to_observation_end) or when r_k is "
                "not observed. Never zero-filled."
            )
        ),
    )


def _future(name: str, description: str, *, dtype: str, shape: str) -> LabelSpec:
    return _speech_grid(
        name,
        description,
        dtype=dtype,
        shape=shape,
        level=Level.GRID,
        time_reference=HORIZON_REF,
        role=Role.TARGET,
        validity=(
            "element null when the horizon window is not entirely observed "
            "(recording end or UNKNOWN before r_k + h): censored, never 'no event'."
        ),
    )


def _segment(
    name: str,
    description: str,
    *,
    segment_type: str,
    columns: tuple[str, ...],
    validity: str,
    source_kind: SourceKind = DET,
    extractor: Extractor = Extractor.SPEECH,
    modalities: tuple[Modality, ...] = (A,),
    requires: tuple[str, ...] = (facts.SPEECH,),
    role: Role = Role.CONTEXT,
    notes: str = "",
) -> LabelSpec:
    return _available(
        name,
        description,
        level=Level.SEGMENT,
        modalities=modalities,
        dtype="table rows",
        shape="one row per interval",
        time_reference=NATIVE_INTERVAL_REF,
        source_kind=source_kind,
        role=role,
        validity=validity,
        extractor=extractor,
        table=Table.SEGMENTS,
        requires=requires,
        columns=columns,
        row_filter=("segment_type", segment_type),
        notes=notes,
    )


def _event(
    name: str,
    description: str,
    *,
    event_type: str,
    columns: tuple[str, ...],
    validity: str,
    source_kind: SourceKind = DET,
    extractor: Extractor = Extractor.SPEECH,
    modalities: tuple[Modality, ...] = (A,),
    requires: tuple[str, ...] = (facts.SPEECH,),
    notes: str = "",
) -> LabelSpec:
    return _available(
        name,
        description,
        level=Level.EVENT,
        modalities=modalities,
        dtype="table rows",
        shape="one row per event",
        time_reference=NATIVE_POINT_REF,
        source_kind=source_kind,
        role=Role.CONTEXT,
        validity=validity,
        extractor=extractor,
        table=Table.EVENTS,
        requires=requires,
        columns=columns,
        row_filter=("event_type", event_type),
        notes=notes,
    )


def _profile(name: str, description: str, columns: tuple[str, ...]) -> LabelSpec:
    return _available(
        name,
        description,
        level=Level.PARTICIPANT,
        modalities=(A,),
        dtype="float64 / int64 columns",
        shape="one row per (recording, participant)",
        time_reference=RECORDING_REF,
        source_kind=DET,
        role=Role.DIAGNOSTIC,
        validity=(
            "null when undefined (e.g. no floor take to average). Computed over the "
            "whole recording, future included: non-causal, never an input for "
            "predicting within the same recording."
        ),
        extractor=Extractor.SPEECH,
        table=Table.PARTICIPANTS,
        columns=columns,
    )


def _metadata(
    name: str,
    description: str,
    *,
    table: Table,
    columns: tuple[str, ...],
    requires: tuple[str, ...] = (),
    validity: str = "never null for a registered recording.",
) -> LabelSpec:
    return _available(
        name,
        description,
        level=Level.RECORDING if table is Table.RECORDINGS else Level.PARTICIPANT,
        modalities=(M,),
        dtype="scalar columns",
        shape="one row per recording"
        if table is Table.RECORDINGS
        else "one row per participant",
        time_reference=RECORDING_REF,
        source_kind=NATIVE,
        role=Role.METADATA,
        validity=validity,
        extractor=Extractor.SPEECH,
        table=table,
        requires=requires,
        columns=columns,
    )


_NOT_DECODED = "null where the cell was not fully decoded."


def _nuisance(
    name: str,
    description: str,
    *,
    extractor: Extractor,
    dtype: str = "float32",
    shape: str = "[]",
    validity: str = _NOT_DECODED,
    columns: tuple[str, ...] | None = None,
) -> LabelSpec:
    audio = extractor is Extractor.AUDIO
    return _available(
        name,
        description,
        level=Level.GRID,
        modalities=(A,) if audio else (V,),
        dtype=dtype,
        shape=shape,
        time_reference=CELL_REF,
        source_kind=DET,
        role=Role.DIAGNOSTIC,
        validity=validity,
        extractor=extractor,
        table=Table.GRID,
        requires=(facts.MEDIA_AUDIO,) if audio else (facts.MEDIA_VIDEO,),
        columns=columns,
        notes="A nuisance/diagnostic control computed on the mixed signal; not a social truth.",
    )


def _social_grid(
    name: str,
    description: str,
    *,
    requires: tuple[str, ...],
    modalities: tuple[Modality, ...],
    validity: str,
    dtype: str = BOOL_PS,
    shape: str = PS,
    columns: tuple[str, ...] | None = None,
    source_kind: SourceKind = NATIVE,
    level: Level = Level.SUBFRAME,
    time_reference: str = SUBFRAME_REF,
) -> LabelSpec:
    return _available(
        name,
        description,
        level=level,
        modalities=modalities,
        dtype=dtype,
        shape=shape,
        time_reference=time_reference,
        source_kind=source_kind,
        role=Role.CONTEXT,
        validity=validity,
        extractor=Extractor.SOCIAL,
        table=Table.GRID,
        requires=requires,
        columns=columns,
    )


_NOT_IMPLEMENTED_VIDEO = (
    "The video model extractor is not implemented: no model has been validated "
    "on egocentric frames of these corpora. The registry entry fixes the contract "
    "so an extractor can be added without changing consumers."
)
_NEEDS_OPERATIONAL_DEFINITION = (
    "Needs an operational definition and an independently validated annotation or "
    "model; no supported corpus provides one, and it is never inferred from "
    "timing alone."
)

INSTANTANEOUS = (
    _speech_grid(
        "instantaneous.speaker_activity_subframes",
        "Whether each participant speaks at any time inside the subframe (native "
        "speech intervals resampled).",
        dtype=BOOL_PS,
        shape=PS,
        source_kind=NATIVE,
        validity=UNKNOWN_NULL,
    ),
    _speech_grid(
        "instantaneous.speaker_activity",
        "Whether each participant speaks at any time inside the decision cell.",
        dtype="list<bool>",
        shape="[participant]",
        source_kind=NATIVE,
        level=Level.GRID,
        time_reference=CELL_REF,
        validity=UNKNOWN_NULL,
    ),
    _speech_grid(
        "instantaneous.ego_speaking_subframes",
        "Whether the camera wearer speaks inside the subframe.",
        dtype=BOOL_S,
        shape=SF,
        source_kind=NATIVE,
        validity=UNKNOWN_NULL,
    ),
    _speech_grid(
        "instantaneous.ego_speaking",
        "Whether the camera wearer speaks inside the decision cell.",
        dtype="bool",
        shape="[]",
        source_kind=NATIVE,
        level=Level.GRID,
        time_reference=CELL_REF,
        validity=UNKNOWN_NULL,
    ),
    _speech_grid(
        "instantaneous.others_active_subframes",
        "Whether any participant other than the wearer speaks inside the subframe.",
        dtype=BOOL_S,
        shape=SF,
        validity=(
            "true as soon as one other participant is known to speak; false only "
            "when every other participant is known silent; null otherwise."
        ),
    ),
    _speech_grid(
        "instantaneous.others_active",
        "Whether any participant other than the wearer speaks inside the cell.",
        dtype="bool",
        shape="[]",
        level=Level.GRID,
        time_reference=CELL_REF,
        validity="three-valued as for others_active_subframes.",
    ),
    _speech_grid(
        "instantaneous.joint_speech_state_subframes",
        "Joint wearer/others state of the subframe: 0 silence, 1 ego only, "
        "2 others only, 3 ego and others.",
        dtype=INT8_S,
        shape=SF,
        validity=UNKNOWN_NULL,
    ),
    _speech_grid(
        "instantaneous.joint_speech_state_occupancy",
        "Fraction of the decision cell spent in each joint state "
        "(silence, ego only, others only, both), from native intervals.",
        dtype="fixed_size_list<float32, 4>",
        shape="[joint_state]",
        level=Level.GRID,
        time_reference=CELL_REF,
        validity="null when any participant is UNKNOWN for part of the cell.",
    ),
    _speech_grid(
        "instantaneous.active_speaker_count_subframes",
        "Number of participants speaking inside the subframe.",
        dtype=INT8_S,
        shape=SF,
        validity=UNKNOWN_NULL,
    ),
    _speech_grid(
        "instantaneous.floor_holder_subframes",
        "Instantaneous unique speaker of the subframe: participant index, -1 NONE "
        "(nobody speaks), -2 MULTIPLE (several speak).",
        dtype=INT16_S,
        shape=SF,
        validity="null = UNKNOWN; -1 and -2 are documented categories, not missing values.",
    ),
    _unsupported(
        "instantaneous.diarized_speech_activity",
        "Speaker activity estimated by an automatic diarization model.",
        level=Level.SUBFRAME,
        modalities=(A,),
        source_kind=EXT,
        external=PYANNOTE,
        reason=(
            "Not implemented: EgoCom and Ego4D provide native speaker-attributed "
            "speech, which a diarization model must never silently replace. Intended "
            "for future corpora without speaker annotations."
        ),
    ),
)

EVENTS = (
    _speech_grid(
        "events.speaker_onset_subframes",
        "Whether each participant starts speaking (SILENT -> SPEAKING) in the subframe.",
        dtype=BOOL_PS,
        shape=PS,
        validity=TRANSITION_NULL,
    ),
    _speech_grid(
        "events.speaker_offset_subframes",
        "Whether each participant stops speaking (SPEAKING -> SILENT) in the subframe.",
        dtype=BOOL_PS,
        shape=PS,
        validity=TRANSITION_NULL,
    ),
    _speech_grid(
        "events.speaker_transition_valid_subframes",
        "Whether an onset/offset of each participant is observable in the subframe.",
        dtype=BOOL_PS,
        shape=PS,
        validity="never null.",
    ),
    _speech_grid(
        "events.ego_onset_subframes",
        "Whether the wearer starts speaking in the subframe.",
        dtype=BOOL_S,
        shape=SF,
        validity=TRANSITION_NULL,
    ),
    _speech_grid(
        "events.ego_offset_subframes",
        "Whether the wearer stops speaking in the subframe.",
        dtype=BOOL_S,
        shape=SF,
        validity=TRANSITION_NULL,
    ),
    _speech_grid(
        "events.other_onset_subframes",
        "Whether any other participant starts speaking in the subframe.",
        dtype=BOOL_S,
        shape=SF,
        validity="true if any other onset is observed; false only if every other "
        "participant's transitions are observable; null otherwise.",
    ),
    _speech_grid(
        "events.other_offset_subframes",
        "Whether any other participant stops speaking in the subframe.",
        dtype=BOOL_S,
        shape=SF,
        validity="three-valued as for other_onset_subframes.",
    ),
    _speech_grid(
        "events.floor_change_subframes",
        "Whether the floor (last unique speaker) passes to another participant in "
        "the subframe.",
        dtype=BOOL_S,
        shape=SF,
        validity="null where the floor is unknown at the subframe start (after "
        "UNKNOWN or the recording start, until someone speaks alone).",
    ),
    _speech_grid(
        "events.previous_floor_holder_subframes",
        "Participant index losing the floor, in subframes holding a floor change.",
        dtype=INT16_S,
        shape=SF,
        validity="null except in subframes holding a floor change (first change).",
    ),
    _speech_grid(
        "events.next_floor_holder_subframes",
        "Participant index gaining the floor, in subframes holding a floor change.",
        dtype=INT16_S,
        shape=SF,
        validity="null except in subframes holding a floor change (last change).",
    ),
    _event(
        "events.speaker_onsets",
        "Every observable onset of every participant, in native time, with its "
        "onset-context primitives and onset type.",
        event_type="onset",
        columns=(
            "participant_index",
            "participant_id",
            "is_ego",
            "previous_unique_speaker_index",
            "was_last_unique_speaker",
            "others_active_before",
            "others_active_at_onset",
            "silence_before_s",
            "simultaneous_other_onset",
            "onset_type",
            "context_valid",
        ),
        validity=(
            "only observable onsets are rows (both sides known). Context columns are "
            "null when unknown; onset_type is 'undetermined' when context_valid is false."
        ),
    ),
    _event(
        "events.speaker_offsets",
        "Every observable offset of every participant, in native time.",
        event_type="offset",
        columns=("participant_index", "participant_id", "is_ego"),
        validity="only observable offsets are rows (both sides known).",
    ),
)

ONSET_CONTEXT = (
    _speech_grid(
        "onset_context.previous_unique_speaker_subframes",
        "At a wearer onset: the last participant who spoke alone before it.",
        dtype=INT16_S,
        shape=SF,
        validity=AT_EGO_ONSET + " Also null when that speaker is unknown.",
    ),
    _speech_grid(
        "onset_context.ego_was_last_unique_speaker_subframes",
        "At a wearer onset: whether the wearer was the last unique speaker.",
        dtype=BOOL_S,
        shape=SF,
        validity=AT_EGO_ONSET,
    ),
    _speech_grid(
        "onset_context.others_active_at_ego_onset_subframes",
        "At a wearer onset: whether another participant speaks at the onset instant.",
        dtype=BOOL_S,
        shape=SF,
        validity=AT_EGO_ONSET,
    ),
    _speech_grid(
        "onset_context.silence_duration_before_ego_onset_subframes",
        "At a wearer onset: duration of the joint silence ending at it (0 when "
        "someone was speaking just before).",
        dtype=FLOAT_S,
        shape=SF,
        validity=AT_EGO_ONSET
        + " Null when the silence starts at the recording start or at UNKNOWN (censored).",
    ),
    _speech_grid(
        "onset_context.simultaneous_other_onset_subframes",
        "At a wearer onset: whether another participant starts within "
        "simultaneous_onset_tolerance_s of it.",
        dtype=BOOL_S,
        shape=SF,
        validity=AT_EGO_ONSET,
    ),
    _speech_grid(
        "onset_context.ego_onset_context_valid_subframes",
        "Whether the subframe holds a wearer onset whose context is fully known.",
        dtype=BOOL_S,
        shape=SF,
        validity="never null.",
    ),
    _speech_grid(
        "onset_context.ego_onset_type_subframes",
        "Type of the wearer onset: 3 overlap (another participant speaks at the "
        "onset instant); otherwise 1 after_silence (the wearer was the last unique "
        "speaker) or 2 floor_transfer (another participant was, including a "
        "zero-length gap); 0 undetermined.",
        dtype=INT8_S,
        shape=SF,
        validity="null except in subframes holding a wearer onset.",
        notes=(
            "mpc-wm names: after_silence = self_resumption, floor_transfer = "
            "floor_take_after_gap, overlap = floor_take_in_overlap. Unlike mpc-wm, "
            "overlap is decided at the onset instant in native time, so a zero-gap "
            "handover is a floor_transfer; the primitives (others active just "
            "before, silence duration, simultaneous onset) are kept alongside."
        ),
    ),
)

OVERLAP = (
    _speech_grid(
        "overlap.within_overlap_subframes",
        "Whether the subframe intersects a within-speaker overlap: the entering "
        "participant stops before the initial one (the initial speaker keeps the "
        "floor).",
        dtype=BOOL_S,
        shape=SF,
        validity=UNKNOWN_NULL + " Also null over overlaps whose type is undetermined.",
    ),
    _speech_grid(
        "overlap.between_overlap_subframes",
        "Whether the subframe intersects a between-speaker overlap: the entering "
        "participant outlasts the initial one.",
        dtype=BOOL_S,
        shape=SF,
        validity=UNKNOWN_NULL + " Also null over overlaps whose type is undetermined.",
    ),
    _speech_grid(
        "overlap.simultaneous_onset_overlap_subframes",
        "Whether the subframe intersects an overlap whose two participants started "
        "within simultaneous_onset_tolerance_s of each other.",
        dtype=BOOL_S,
        shape=SF,
        validity=UNKNOWN_NULL,
    ),
    _segment(
        "overlap.overlap_events",
        "Every pairwise overlap between an initial speech run and a run of another "
        "participant starting inside it: start (entry), end (first of the two "
        "ends), duration, participants, type and previous floor holder.",
        segment_type="overlap",
        columns=(
            "initial_participant_index",
            "entering_participant_index",
            "overlap_type",
            "previous_floor_holder_index",
            "duration_s",
            "valid",
        ),
        validity=(
            "valid=false (type 'undetermined') when a run end that decides the type "
            "is censored by UNKNOWN or the recording edge."
        ),
        notes="Heldner & Edlund (2010) within/between overlap typing, pairwise as in mpc-wm.",
    ),
    _unsupported(
        "overlap.overlap_function",
        "Pragmatic function of an overlap: competitive, collaborative or "
        "backchannel-like.",
        level=Level.SEGMENT,
        modalities=(A, T),
        source_kind=HUMAN,
        reason=(
            "A pragmatic interpretation; never a consequence of timing. Needs human "
            "annotation or a validated model with recorded provenance."
        ),
    ),
)

TIMING = (
    _timing(
        "timing.time_since_ego_onset",
        "Seconds since the last wearer onset before r_k.",
        target=False,
    ),
    _timing(
        "timing.time_since_ego_offset",
        "Seconds since the last wearer offset before r_k.",
        target=False,
    ),
    _timing(
        "timing.time_since_other_onset",
        "Seconds since the last onset of any other participant before r_k.",
        target=False,
    ),
    _timing(
        "timing.time_since_other_offset",
        "Seconds since the last offset of any other participant before r_k.",
        target=False,
    ),
    _timing(
        "timing.time_since_speaker_activity",
        "Seconds since any participant last spoke (0 when someone speaks just "
        "before r_k).",
        target=False,
    ),
    _timing(
        "timing.time_since_floor_change",
        "Seconds since the last floor change before r_k.",
        target=False,
    ),
    _timing(
        "timing.time_to_next_ego_onset",
        "Seconds from r_k to the next wearer onset.",
        target=True,
    ),
    _timing(
        "timing.time_to_next_ego_offset",
        "Seconds from r_k to the next wearer offset.",
        target=True,
    ),
    _timing(
        "timing.time_to_next_other_onset",
        "Seconds from r_k to the next onset of any other participant.",
        target=True,
    ),
    _timing(
        "timing.time_to_next_other_offset",
        "Seconds from r_k to the next offset of any other participant.",
        target=True,
    ),
    _timing(
        "timing.time_to_next_floor_change",
        "Seconds from r_k to the next floor change.",
        target=True,
    ),
    _timing(
        "timing.time_to_next_speaker_onset",
        "Seconds from r_k to the next onset of any participant.",
        target=True,
    ),
    _speech_grid(
        "timing.silence_duration",
        "Elapsed duration of the joint silence in progress at r_k.",
        dtype="float32 seconds (+ bool valid)",
        shape="[]",
        level=Level.GRID,
        time_reference=CELL_END_REF,
        columns=("silence_duration", "silence_duration_valid"),
        validity=(
            "null when nobody is silent-jointly at r_k or when the silence started "
            "at the recording start or UNKNOWN (censored)."
        ),
    ),
    _speech_grid(
        "timing.observation_bounds",
        "Distance from r_k to the start and to the end of the observed span "
        "containing it: the censoring bounds of every time_since/time_to label.",
        dtype="float32 seconds",
        shape="[]",
        level=Level.GRID,
        time_reference=CELL_END_REF,
        role=Role.METADATA,
        columns=("time_since_observation_start", "time_to_observation_end"),
        validity="null when r_k is not observed.",
    ),
    _segment(
        "timing.silences",
        "Every joint silence: pause (the same participant speaks before and after) "
        "or gap (a different participant), with duration.",
        segment_type="silence",
        columns=(
            "silence_type",
            "previous_speaker_index",
            "next_speaker_index",
            "duration_s",
            "valid",
        ),
        validity=(
            "valid=false and silence_type 'undetermined' when bounded by UNKNOWN or "
            "the recording edge, or when several participants stop/start at its ends."
        ),
    ),
)

TURNS = (
    _segment(
        "turns.speech_runs",
        "Maximal continuous speech of one participant (native intervals unioned, "
        "no pause closure).",
        segment_type="speech_run",
        columns=(
            "participant_index",
            "participant_id",
            "is_ego",
            "duration_s",
            "start_censored",
            "end_censored",
            "valid",
        ),
        validity="valid=false when either end touches UNKNOWN or the recording edge.",
    ),
    _segment(
        "turns.turns",
        "Speech runs of one participant joined across internal pauses of at most "
        "turn_max_internal_pause_s (versioned rule), with pause statistics and the "
        "neighbouring turns' speakers.",
        segment_type="turn",
        columns=(
            "participant_index",
            "participant_id",
            "is_ego",
            "duration_s",
            "run_count",
            "pause_count",
            "pause_total_s",
            "pause_max_s",
            "previous_turn_participant_index",
            "next_turn_participant_index",
            "fto_from_previous_s",
            "start_censored",
            "end_censored",
            "valid",
        ),
        validity=(
            "valid=false when either end is censored. Neighbour columns are null at "
            "the recording edges; fto_from_previous_s is null when the previous turn "
            "has the same speaker."
        ),
    ),
    _event(
        "turns.floor_transfers",
        "Every change of the last unique speaker from one participant to another, "
        "with the floor-transfer offset FTO = (start of the new holder's run) - "
        "(end of the previous holder's last run): negative = overlap, positive = gap.",
        event_type="floor_change",
        columns=(
            "previous_holder_index",
            "next_holder_index",
            "fto_s",
            "transfer_kind",
            "locality",
            "previous_run_end_s",
            "next_run_start_s",
        ),
        validity=(
            "only transfers between two known holders are rows. locality marks FTOs "
            "beyond max_local_gap_s / max_local_overlap_s (kept, not dropped)."
        ),
    ),
)

NEXT_SPEAKER = tuple(
    _speech_grid(
        f"next_speaker.{column}",
        description,
        dtype=dtype,
        shape="[]",
        level=Level.GRID,
        time_reference=CELL_END_REF,
        role=Role.TARGET,
        columns=(column, f"{column}_valid"),
        validity=validity,
    )
    for column, description, dtype, validity in (
        (
            "next_speaker",
            "Participant index of the next onset at or after r_k.",
            "int16 (+ bool valid)",
            "null when censored before any onset, or when several participants tie.",
        ),
        (
            "next_unique_speaker",
            (
                "Participant index of the next participant to start speaking alone "
                "at or after r_k (may be the current speaker resuming)."
            ),
            "int16 (+ bool valid)",
            "null when censored.",
        ),
        (
            "current_speaker_continues",
            "Whether the next unique speaker is the current last unique speaker.",
            "bool (+ bool valid)",
            (
                "null when the current last unique speaker is unknown or the next "
                "one is censored."
            ),
        ),
        (
            "floor_transfer_target",
            (
                "Participant index gaining the floor at the next floor change at "
                "or after r_k."
            ),
            "int16 (+ bool valid)",
            "null when no floor change is observed before the end of the observed span.",
        ),
    )
)

FUTURE = (
    _future(
        "future.future_speaker_activity",
        "Whether each participant speaks within each horizon.",
        dtype="fixed_size_list<list<bool>, H>",
        shape="[horizon, participant]",
    ),
    _future(
        "future.future_ego_activity",
        "Whether the wearer speaks within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_others_activity",
        "Whether any other participant speaks within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_joint_speech_state",
        "Joint state (0-3, as instantaneous.joint_speech_state_subframes) just "
        "before r_k + h.",
        dtype="fixed_size_list<int8, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_floor_holder",
        "Instantaneous floor holder (index, -1 NONE, -2 MULTIPLE) just before r_k + h.",
        dtype="fixed_size_list<int16, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_overlap",
        "Whether two or more participants speak simultaneously within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_ego_onset",
        "Whether the wearer starts speaking within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_ego_offset",
        "Whether the wearer stops speaking within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_other_onset",
        "Whether any other participant starts speaking within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_other_offset",
        "Whether any other participant stops speaking within each horizon.",
        dtype="fixed_size_list<bool, H>",
        shape="[horizon]",
    ),
    _future(
        "future.future_next_speaker",
        "Participant index of the first onset within each horizon, -1 NOBODY when "
        "nobody starts within it.",
        dtype="fixed_size_list<int16, H>",
        shape="[horizon]",
    ),
)

_EGO_SOLO = (
    "computed only on audio frames where the wearer is the sole known speaker "
    "(native annotations), on the wearer's own camera microphone."
)
PROSODY = (
    _available(
        "prosody.ego_f0_hz",
        "Median fundamental frequency of the wearer's voiced frames in the cell, "
        + _EGO_SOLO,
        level=Level.GRID,
        modalities=(A,),
        dtype="float32 Hz",
        shape="[]",
        time_reference=CELL_REF,
        source_kind=EXT,
        role=Role.CONTEXT,
        validity=(
            "null when the ego-solo mask covers less than min_mask_fraction of the "
            "cell, or when no masked frame is voiced. Overlap is always masked."
        ),
        extractor=Extractor.AUDIO,
        table=Table.GRID,
        requires=(facts.SPEECH, facts.MEDIA_AUDIO),
        external=PRAAT,
    ),
    _available(
        "prosody.ego_voiced_fraction",
        "Fraction of the masked frames Praat judges voiced, " + _EGO_SOLO,
        level=Level.GRID,
        modalities=(A,),
        dtype="float32",
        shape="[]",
        time_reference=CELL_REF,
        source_kind=EXT,
        role=Role.CONTEXT,
        validity="null when the ego-solo mask covers less than min_mask_fraction.",
        extractor=Extractor.AUDIO,
        table=Table.GRID,
        requires=(facts.SPEECH, facts.MEDIA_AUDIO),
        external=PRAAT,
    ),
    _available(
        "prosody.ego_rms_db",
        "RMS energy (dBFS) of the wearer's speech in the cell, " + _EGO_SOLO,
        level=Level.GRID,
        modalities=(A,),
        dtype="float32 dBFS",
        shape="[]",
        time_reference=CELL_REF,
        source_kind=DET,
        role=Role.CONTEXT,
        validity="null when the ego-solo mask covers less than min_mask_fraction.",
        extractor=Extractor.AUDIO,
        table=Table.GRID,
        requires=(facts.SPEECH, facts.MEDIA_AUDIO),
    ),
    _available(
        "prosody.ego_solo_mask_fraction",
        "Fraction of the cell where the wearer is the sole known speaker: the "
        "attribution mask of every ego prosody label.",
        level=Level.GRID,
        modalities=(A,),
        dtype="float32",
        shape="[]",
        time_reference=CELL_REF,
        source_kind=DET,
        role=Role.DIAGNOSTIC,
        validity="null where any participant is UNKNOWN in the cell.",
        extractor=Extractor.AUDIO,
        table=Table.GRID,
        requires=(facts.SPEECH, facts.MEDIA_AUDIO),
    ),
    _unsupported(
        "prosody.others_f0",
        "F0 of participants other than the wearer.",
        level=Level.GRID,
        modalities=(A,),
        source_kind=EXT,
        external=PRAAT,
        reason=(
            "No validated attribution: other participants are far-field on the "
            "wearer's microphone, and cross-recording use of their own camera "
            "microphone is not validated."
        ),
    ),
    _unsupported(
        "prosody.egemaps",
        "eGeMAPS low-level descriptors (loudness, spectral flux, MFCC 1-4, jitter, "
        "shimmer, HNR, formants, voiced segments).",
        level=Level.GRID,
        modalities=(A,),
        source_kind=EXT,
        external=OPENSMILE,
        reason=(
            "Not implemented: openSMILE's licence restricts use to research and "
            "education and needs review before redistributing derived features."
        ),
    ),
    _unsupported(
        "prosody.mfcc",
        "Mel-frequency cepstral coefficients.",
        level=Level.GRID,
        modalities=(A,),
        source_kind=DET,
        reason=(
            "Not implemented: no downstream use justifies it yet; learned audio "
            "encoders already consume the waveform."
        ),
    ),
    _unsupported(
        "prosody.voice_quality",
        "Jitter, shimmer and harmonics-to-noise ratio of the wearer's voice.",
        level=Level.GRID,
        modalities=(A,),
        source_kind=EXT,
        external=PRAAT,
        reason=(
            "Not validated: these measures are unreliable on far-field, noisy "
            "egocentric recordings without sustained vowels."
        ),
    ),
    _unsupported(
        "prosody.final_lengthening",
        "Lengthening of the last syllable/phone of a turn.",
        level=Level.SEGMENT,
        modalities=(A, T),
        source_kind=EXT,
        reason="Needs phone-level forced alignment, which no supported corpus provides.",
    ),
    _unsupported(
        "prosody.perceptual_loudness",
        "Perceptual loudness (e.g. EBU R128 / Zwicker).",
        level=Level.GRID,
        modalities=(A,),
        source_kind=DET,
        reason="Not implemented: prosody.ego_rms_db covers energy/intensity.",
    ),
)

_TEXT_UNIT_VALIDITY = (
    "one row per text unit: a turn with word timings (unit='turn') or a native "
    "utterance (unit='utterance'). Columns are null when the unit has no lexical token."
)
TEXT = (
    _event(
        "text.tokens",
        "Native transcript tokens (words, or utterances when the corpus only times "
        "utterances) with speaker, timing, token class and turn association.",
        event_type="token",
        columns=(
            "participant_index",
            "participant_id",
            "unit",
            "text",
            "token_class",
            "end_s",
            "timing_valid",
            "turn_id",
            "source_row",
        ),
        validity=(
            "untimed tokens are kept with null time_s/end_s and timing_valid=false; "
            "turn_id is null when the token cannot be placed in a turn."
        ),
        source_kind=NATIVE,
        extractor=Extractor.TEXT,
        modalities=(T,),
        requires=(facts.SPEECH, facts.TRANSCRIPT),
    ),
    _segment(
        "text.speech_rate",
        "Lexical words per second of each turn (per turn duration and per voiced "
        "second), from word-level timings.",
        segment_type="text_unit",
        columns=(
            "unit",
            "word_count",
            "timed_word_count",
            "words_per_second",
            "words_per_voiced_second",
            "speech_rate_valid",
        ),
        validity="speech_rate_valid=false for utterance units or turns without lexical words.",
        extractor=Extractor.TEXT,
        modalities=(T, A),
        requires=(facts.SPEECH, facts.WORDS),
    ),
    _segment(
        "text.final_punctuation",
        "Punctuation mark closing the text unit as transcribed ('.', '?', '!', ',').",
        segment_type="text_unit",
        columns=("final_punctuation",),
        validity=_TEXT_UNIT_VALIDITY + " Null when the unit ends without punctuation.",
        extractor=Extractor.TEXT,
        modalities=(T,),
        requires=(facts.SPEECH, facts.TRANSCRIPT),
    ),
    _segment(
        "text.interrogative_cue",
        "Whether the text unit ends with a transcribed question mark.",
        segment_type="text_unit",
        columns=("interrogative_cue",),
        validity=_TEXT_UNIT_VALIDITY
        + " A transcription cue, not a dialogue-act label.",
        extractor=Extractor.TEXT,
        modalities=(T,),
        requires=(facts.SPEECH, facts.TRANSCRIPT),
    ),
    _segment(
        "text.turn_final_token",
        "Last lexical token of the text unit, lower-cased.",
        segment_type="text_unit",
        columns=("final_token",),
        validity=_TEXT_UNIT_VALIDITY,
        extractor=Extractor.TEXT,
        modalities=(T,),
        requires=(facts.SPEECH, facts.TRANSCRIPT),
    ),
    _segment(
        "text.discourse_markers",
        "Discourse marker opening / closing the text unit, from a fixed versioned "
        "lexicon (e.g. 'so', 'well', 'you know').",
        segment_type="text_unit",
        columns=("initial_discourse_marker", "final_discourse_marker"),
        validity=_TEXT_UNIT_VALIDITY + " Null when no lexicon entry matches.",
        extractor=Extractor.TEXT,
        modalities=(T,),
        requires=(facts.SPEECH, facts.TRANSCRIPT),
    ),
    *(
        _unsupported(
            f"text.{name}",
            description,
            level=Level.SEGMENT,
            modalities=(T,),
            source_kind=kind,
            reason=reason,
        )
        for name, description, kind, reason in (
            (
                "lexical_completion",
                "Whether the words so far form a lexically complete unit.",
                EXT,
                "Needs a validated language model; not inferred from punctuation.",
            ),
            (
                "syntactic_completion",
                "Whether the words so far form a syntactically complete unit.",
                EXT,
                "Needs a validated parser/model; not inferred from punctuation.",
            ),
            (
                "semantic_completion",
                "Whether the utterance is semantically complete.",
                HUMAN,
                _NEEDS_OPERATIONAL_DEFINITION,
            ),
            (
                "dialogue_act",
                "Dialogue act of the utterance.",
                HUMAN,
                _NEEDS_OPERATIONAL_DEFINITION,
            ),
            (
                "adjacency_pair_role",
                "First/second pair part role of the utterance.",
                HUMAN,
                _NEEDS_OPERATIONAL_DEFINITION,
            ),
            (
                "agreement",
                "Agreement / disagreement expressed by the utterance.",
                HUMAN,
                _NEEDS_OPERATIONAL_DEFINITION,
            ),
            (
                "repair",
                "Self- or other-initiated repair.",
                HUMAN,
                _NEEDS_OPERATIONAL_DEFINITION,
            ),
            (
                "relevance",
                "Relevance of the utterance to the previous one.",
                HUMAN,
                _NEEDS_OPERATIONAL_DEFINITION,
            ),
            ("common_ground", "Grounding acts.", HUMAN, _NEEDS_OPERATIONAL_DEFINITION),
        )
    ),
    _unsupported(
        "text.asr_transcript",
        "Automatic transcript with word-level alignment and diarization.",
        level=Level.EVENT,
        modalities=(A, T),
        source_kind=EXT,
        external=WHISPERX,
        reason=(
            "Not implemented: EgoCom and Ego4D ship native transcripts that ASR must "
            "never silently replace. Intended for future corpora without transcripts."
        ),
    ),
)

VIDEO = tuple(
    _unsupported(
        f"video.{name}",
        description,
        level=Level.SUBFRAME,
        modalities=(V,),
        source_kind=EXT,
        external=tool,
        reason=_NOT_IMPLEMENTED_VIDEO + extra,
    )
    for name, description, tool, extra in (
        ("face_visible", "Whether a face is detected in the frame.", MEDIAPIPE, ""),
        (
            "face_bbox",
            "Detected face bounding boxes.",
            MEDIAPIPE,
            " Ego4D's native boxes are social_native.face_track_bbox.",
        ),
        ("face_landmarks", "Dense facial landmarks.", MEDIAPIPE, ""),
        (
            "head_pose",
            "Head yaw/pitch/roll from the facial transformation.",
            MEDIAPIPE,
            "",
        ),
        (
            "gaze_proxy",
            "Gaze direction estimate.",
            MEDIAPIPE,
            " A head pose is never relabelled as gaze.",
        ),
        (
            "gaze_target",
            "Which participant a person looks at.",
            MEDIAPIPE,
            " Ego4D's native Looking-At-Me is social_native.looking_at_wearer_subframes.",
        ),
        ("mouth_openness", "Lip distance normalized by face size.", MEDIAPIPE, ""),
        ("nod_shake", "Head nod / shake gestures.", MEDIAPIPE, ""),
        ("facial_movement", "Facial motion energy.", MEDIAPIPE, ""),
        ("upper_body_pose", "Upper-body keypoints.", MMPOSE, ""),
        (
            "body_orientation",
            "Torso orientation.",
            MMPOSE,
            " An orientation is never relabelled as an addressee.",
        ),
        ("hand_pose", "Hand and arm keypoints.", MMPOSE, ""),
        ("gesture_activity", "Gesture motion energy.", MMPOSE, ""),
        ("pre_speech_movement", "Body movement preceding an onset.", MMPOSE, ""),
        ("participant_geometry", "Relative positions of participants.", MMPOSE, ""),
        (
            "camera_motion",
            "Estimated ego-motion of the head-mounted camera.",
            MEDIAPIPE,
            " A frame-difference proxy is nuisance.frame_difference.",
        ),
    )
)

_LAM = (
    "Ego4D Social Looking-At-Me: frame-level label of a tracked, identified face "
    "being directed at the camera wearer."
)
_TTM = (
    "Ego4D Social Talking-To-Me: utterance-level label of a speaking, tracked "
    "person talking to the camera wearer."
)
SOCIAL_NATIVE = (
    _social_grid(
        "social_native.looking_at_wearer_subframes",
        _LAM + " True inside a positive native segment of the participant.",
        requires=(facts.SPEECH, facts.SOCIAL_LOOKING, facts.FACE_TRACKS),
        modalities=(V,),
        columns=("looking_at_wearer_subframes", "looking_at_wearer_valid_subframes"),
        validity=(
            "false where the participant's face is tracked (or a native negative "
            "segment covers it) and no positive segment does; null (valid=false) "
            "where the face is not tracked: not annotated is never negative."
        ),
    ),
    _social_grid(
        "social_native.talking_to_wearer_subframes",
        _TTM + " Value of the native talking segment covering the subframe.",
        requires=(facts.SPEECH, facts.SOCIAL_TALKING),
        modalities=(A, V),
        columns=("talking_to_wearer_subframes", "talking_to_wearer_valid_subframes"),
        validity="null (valid=false) outside native talking segments of the participant.",
    ),
    _social_grid(
        "social_native.anyone_looking_at_wearer_subframes",
        "Whether any relevant participant looks at the wearer (tri-state aggregate "
        "of looking_at_wearer_subframes).",
        requires=(facts.SPEECH, facts.SOCIAL_LOOKING, facts.FACE_TRACKS),
        modalities=(V,),
        dtype=BOOL_S,
        shape=SF,
        source_kind=DET,
        columns=(
            "anyone_looking_at_wearer_subframes",
            "anyone_looking_at_wearer_valid_subframes",
        ),
        validity=(
            "true if any participant is true; false only when at least one relevant "
            "participant (tracked or covered) exists and every relevant one is known "
            "false; null otherwise."
        ),
    ),
    _social_grid(
        "social_native.anyone_talking_to_wearer_subframes",
        "Whether any relevant participant talks to the wearer (tri-state aggregate).",
        requires=(facts.SPEECH, facts.SOCIAL_TALKING),
        modalities=(A, V),
        dtype=BOOL_S,
        shape=SF,
        source_kind=DET,
        columns=(
            "anyone_talking_to_wearer_subframes",
            "anyone_talking_to_wearer_valid_subframes",
        ),
        validity=(
            "relevant = covered by a talking segment or speaking; true if any is "
            "true; false only when every relevant participant is known false; null "
            "otherwise (an active speaker without a segment hides a possible positive)."
        ),
    ),
    _social_grid(
        "social_native.face_tracked_subframes",
        "Whether a native face-track box of the participant exists in the subframe "
        "(frames mapped to the subframe containing their start time).",
        requires=(facts.SPEECH, facts.FACE_TRACKS),
        modalities=(V,),
        validity="null only where the recording is UNKNOWN.",
    ),
    _social_grid(
        "social_native.face_track_bbox",
        "Mean native face box (x, y, width, height in source pixels) of each "
        "participant over its tracked frames in the cell.",
        requires=(facts.SPEECH, facts.FACE_TRACKS),
        modalities=(V,),
        dtype="list<fixed_size_list<float32, 4>>",
        shape="[participant, box]",
        level=Level.GRID,
        time_reference=CELL_REF,
        validity="null for a participant without a tracked frame in the cell.",
    ),
    _segment(
        "social_native.social_segments",
        "Native LAM/TTM segments exactly as released: person (tracking) id, kind, "
        "is_at_me, raw annotation target, frames, and whether the person resolves "
        "to a participant (unresolved tracks are kept).",
        segment_type="social_segment",
        columns=(
            "participant_index",
            "participant_id",
            "person",
            "kind",
            "is_at_me",
            "annotation_target",
            "resolved",
            "start_frame",
            "end_frame",
            "duration_s",
        ),
        validity="participant_index is null for unresolved persons (e.g. '-1').",
        source_kind=NATIVE,
        extractor=Extractor.SOCIAL,
        modalities=(A, V),
        requires=(facts.SPEECH, facts.SOCIAL_LOOKING, facts.SOCIAL_TALKING),
        notes=(
            "annotation_target is the release's raw 'target' field; its semantics are "
            "undocumented, so it is preserved and never promoted to an addressee."
        ),
    ),
    _segment(
        "social_native.face_tracks",
        "One row per native face track: participant, tracking id, first/last frame "
        "and frame count.",
        segment_type="face_track",
        columns=(
            "participant_index",
            "participant_id",
            "track_id",
            "first_frame",
            "last_frame",
            "frame_count",
            "duration_s",
        ),
        validity="never null.",
        source_kind=NATIVE,
        extractor=Extractor.SOCIAL,
        modalities=(V,),
        requires=(facts.SPEECH, facts.FACE_TRACKS),
    ),
)

ADDRESSEE = (
    _unsupported(
        "addressee.addressee",
        "Participant(s) an utterance is addressed to, 'broadcast' or 'unknown', "
        "with validity and source kind.",
        level=Level.SEGMENT,
        modalities=(A, V, T),
        source_kind=HUMAN,
        reason=(
            "No supported corpus annotates addressees. Ego4D TTM only states whether "
            "the wearer is addressed (social_native.talking_to_wearer_subframes); it "
            "is never generalized. A future model may provide it as external_model."
        ),
    ),
    _unsupported(
        "addressee.broadcast",
        "Whether an utterance addresses the whole group.",
        level=Level.SEGMENT,
        modalities=(A, V, T),
        source_kind=HUMAN,
        reason="No supported corpus annotates it.",
    ),
)

BACKCHANNEL = (
    _unsupported(
        "backchannel.events",
        "Backchannel events: speaker, recipient, onset/end, vocal/non-vocal, "
        "latency, validity.",
        level=Level.EVENT,
        modalities=(A, V, T),
        source_kind=HUMAN,
        reason=(
            "Never inferred from duration: a short utterance is not a backchannel by "
            "definition. Needs annotation or a validated method."
        ),
    ),
)

PROFILES = (
    _profile(
        "profiles.speaking_time",
        "Total speaking time, observed time and speaking ratio of the participant.",
        ("speaking_time_s", "observed_duration_s", "speaking_ratio"),
    ),
    _profile(
        "profiles.turn_statistics",
        "Speech-run and turn counts, turns per minute and duration statistics.",
        (
            "speech_run_count",
            "run_duration_mean_s",
            "run_duration_median_s",
            "turn_count",
            "turns_per_minute",
            "turn_duration_mean_s",
            "turn_duration_median_s",
            "turn_duration_std_s",
        ),
    ),
    _profile(
        "profiles.pause_statistics",
        "Within-speaker pauses (silences typed 'pause' after this participant).",
        ("pause_count", "pause_duration_mean_s", "pause_duration_median_s"),
    ),
    _profile(
        "profiles.onset_offset_counts",
        "Observable onsets and offsets, and onsets by type.",
        (
            "onset_count",
            "offset_count",
            "onset_after_silence_count",
            "onset_floor_transfer_count",
            "onset_overlap_count",
        ),
    ),
    _profile(
        "profiles.overlap_statistics",
        "Overlaps entered and received, by type; time in overlap; "
        "interruption-like count = between-speaker overlaps entered (an observable "
        "timing pattern, not a judgement of interruption).",
        (
            "overlaps_entered",
            "overlaps_received",
            "within_overlaps_entered",
            "between_overlaps_entered",
            "interruption_like_count",
            "overlap_time_s",
        ),
    ),
    _profile(
        "profiles.floor_transfer_statistics",
        "Floor takes and yields, FTO distribution of takes and response latency "
        "(positive FTO) statistics.",
        (
            "floor_takes",
            "floor_yields",
            "fto_take_mean_s",
            "fto_take_median_s",
            "fto_take_q10_s",
            "fto_take_q90_s",
            "response_latency_mean_s",
            "response_latency_median_s",
        ),
    ),
    _profile(
        "profiles.interaction_profile_features",
        "Every numeric profile statistic as one vector, in the order of "
        "profiles.interaction_profile_feature_names.",
        ("interaction_profile_features",),
    ),
    _profile(
        "profiles.interaction_profile_feature_names",
        "Names of the interaction_profile_features vector entries.",
        ("interaction_profile_feature_names",),
    ),
)

SOCIAL_STATES = tuple(
    _unsupported(
        f"social_states.{name}",
        description,
        level=Level.SEGMENT,
        modalities=(A, V, T),
        source_kind=HUMAN,
        reason=_NEEDS_OPERATIONAL_DEFINITION,
    )
    for name, description in (
        ("engagement", "Participant engagement."),
        ("dominance", "Conversational dominance."),
        ("leadership", "Emergent leadership."),
        ("rapport", "Rapport between participants."),
        ("cohesion", "Group cohesion."),
        ("tension", "Interpersonal tension."),
        ("awkwardness", "Perceived awkwardness."),
        ("stance", "Stance taken by a participant."),
        ("affect_emotion", "Affect or emotion expressed."),
        ("agreement_conflict", "Agreement or conflict between participants."),
        ("floor_partition", "Subgroups holding separate floors (schisming)."),
    )
)

_AUDIO_NUISANCE = (
    ("global_audio_rms", "RMS amplitude of the mixed audio in the cell.", None),
    ("zero_crossing_rate", "Zero-crossing rate of the mixed audio in the cell.", None),
    ("spectral_centroid", "Spectral centroid (Hz) of the Hann-windowed cell.", None),
    ("spectral_bandwidth", "Spectral bandwidth (Hz) around the centroid.", None),
    (
        "spectral_flux",
        (
            "L2 distance between the normalized magnitude spectra of this cell and "
            "the previous one."
        ),
        "null for the first decoded cell and where either cell is not decoded or silent.",
    ),
    (
        "spectral_valid",
        "Whether the cell had non-zero spectral energy (spectral features defined).",
        "false where the cell is digital silence or not decoded.",
    ),
    (
        "background_noise_proxy",
        "20th percentile of the absolute sample amplitude in the cell.",
        None,
    ),
    ("audio_valid", "Whether the whole cell was decoded.", "never null."),
)
_VIDEO_NUISANCE = (
    (
        "frame_mean_rgb",
        "Mean RGB of the cell's frames.",
        "fixed_size_list<float32, 3>",
        "[rgb]",
    ),
    ("brightness", "Mean Rec.709 luma of the cell's frames.", "float32", "[]"),
    ("contrast", "Standard deviation of pixel values.", "float32", "[]"),
    ("saturation", "Mean HSV saturation.", "float32", "[]"),
    ("blur_score", "Variance of the Laplacian (low = blurry).", "float32", "[]"),
    (
        "dominant_colour",
        "Centre of the most frequent 4-level-per-channel RGB bin.",
        "fixed_size_list<float32, 3>",
        "[rgb]",
    ),
    (
        "frame_difference",
        (
            "Mean absolute difference between consecutive frames (motion proxy, not "
            "an ego-motion estimate)."
        ),
        "float32",
        "[]",
    ),
    ("frame_valid", "Whether every frame of the cell was decoded.", "bool", "[]"),
)
NUISANCE = (
    *(
        _nuisance(
            f"nuisance.{name}",
            description,
            extractor=Extractor.AUDIO,
            dtype="bool" if name.endswith("valid") else "float32",
            validity=validity or _NOT_DECODED,
        )
        for name, description, validity in _AUDIO_NUISANCE
    ),
    *(
        _nuisance(
            f"nuisance.{name}",
            description,
            extractor=Extractor.VIDEO,
            dtype=dtype,
            shape=shape,
            validity="never null."
            if name == "frame_valid"
            else "null where the cell was not fully decoded.",
        )
        for name, description, dtype, shape in _VIDEO_NUISANCE
    ),
)

METADATA = (
    _metadata(
        "metadata.recording_identity",
        "Dataset, conversation, view (camera) and wearer identifiers.",
        table=Table.RECORDINGS,
        columns=("dataset", "conversation_id", "view_id", "wearer_id"),
    ),
    _metadata(
        "metadata.participants",
        "Participant ids in canonical order (wearer first), their count and the "
        "wearer's index (always 0).",
        table=Table.RECORDINGS,
        columns=("participant_ids", "participant_count", "wearer_index"),
    ),
    _metadata(
        "metadata.recording_span",
        "Start, end and duration of the recording on its canonical clock.",
        table=Table.RECORDINGS,
        columns=("start_s", "end_s", "duration_s"),
    ),
    _metadata(
        "metadata.source_offset",
        "Position (seconds) of the recording inside its longer source video.",
        table=Table.RECORDINGS,
        columns=("source_offset_s",),
        requires=(facts.META_SOURCE_OFFSET,),
    ),
    _metadata(
        "metadata.background_conditions",
        "Native background fan / music flags.",
        table=Table.RECORDINGS,
        columns=("background_fan", "background_music"),
        requires=(facts.META_BACKGROUND,),
        validity="null when the corpus leaves the flag empty.",
    ),
    _metadata(
        "metadata.participant_identity",
        "Whether the participant is the wearer and resolves to an annotated person.",
        table=Table.PARTICIPANTS,
        columns=("is_ego", "is_resolved"),
    ),
    _metadata(
        "metadata.participant_native_speaker",
        "Native first-language flag of the participant.",
        table=Table.PARTICIPANTS,
        columns=("native_speaker",),
        requires=(facts.META_NATIVE_SPEAKER,),
        validity="null when the corpus does not state it for this participant.",
    ),
    _metadata(
        "metadata.participant_is_host",
        "Native host flag of the participant.",
        table=Table.PARTICIPANTS,
        columns=("is_host",),
        requires=(facts.META_HOST,),
        validity="null when the corpus does not state it for this participant.",
    ),
)

REGISTRY: tuple[LabelSpec, ...] = (
    *INSTANTANEOUS,
    *EVENTS,
    *ONSET_CONTEXT,
    *OVERLAP,
    *TIMING,
    *TURNS,
    *NEXT_SPEAKER,
    *FUTURE,
    *PROSODY,
    *TEXT,
    *VIDEO,
    *SOCIAL_NATIVE,
    *ADDRESSEE,
    *BACKCHANNEL,
    *PROFILES,
    *SOCIAL_STATES,
    *NUISANCE,
    *METADATA,
)
"""Every label, in documentation order."""

validate(REGISTRY)


def extractor_labels(extractor: Extractor) -> tuple[LabelSpec, ...]:
    """Available labels materialized by ``extractor``."""
    return tuple(spec for spec in REGISTRY if spec.extractor is extractor)


def label_counts() -> dict[str, Any]:
    """Headline counts of the catalog, for reports."""
    counts: dict[str, Any] = {
        "labels": len(REGISTRY),
        "available": sum(spec.extractor is not None for spec in REGISTRY),
        "by_source_kind": {},
        "by_family": {},
    }
    for spec in REGISTRY:
        key = str(spec.source_kind)
        counts["by_source_kind"][key] = counts["by_source_kind"].get(key, 0) + 1
        counts["by_family"][spec.family] = counts["by_family"].get(spec.family, 0) + 1
    return counts


__all__ = ["REGISTRY", "extractor_labels", "label_counts"]
