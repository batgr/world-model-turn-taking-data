"""The registry contract: every label is described once, completely and consistently."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from conv_wm.data.labels import facts
from conv_wm.data.labels.catalog import REGISTRY, label_counts
from conv_wm.data.labels.registry import (
    FAMILIES,
    LABEL_SCHEMA_VERSION,
    REGISTRY_VERSION,
    TABLE_KEYS,
    Availability,
    Extractor,
    LabelSpec,
    Level,
    Modality,
    RegistryError,
    Role,
    SourceKind,
    Table,
    registry_document,
    support_matrix,
    validate,
)

DOCUMENTED_FIELDS = {
    "name",
    "family",
    "description",
    "level",
    "modalities",
    "dtype",
    "shape",
    "time_reference",
    "source_kind",
    "role",
    "validity",
    "version",
    "availability",
    "requires",
    "extractor",
    "table",
    "columns",
    "row_filter",
    "external",
    "unsupported_reason",
    "notes",
}


def test_names_are_unique_and_prefixed_by_a_known_family():
    names = [spec.name for spec in REGISTRY]
    assert len(names) == len(set(names))
    for spec in REGISTRY:
        assert spec.name.split(".")[0] == spec.family
        assert spec.family in FAMILIES


def test_every_enum_field_holds_a_valid_value():
    for spec in REGISTRY:
        assert spec.level in Level
        assert spec.source_kind in SourceKind
        assert spec.role in Role
        assert spec.modalities and all(m in Modality for m in spec.modalities)


def test_every_family_of_the_specification_is_present():
    assert {spec.family for spec in REGISTRY} == set(FAMILIES)


def test_external_labels_document_their_tool_and_licence():
    for spec in REGISTRY:
        if spec.source_kind is SourceKind.EXTERNAL_MODEL:
            assert spec.external is not None
            assert spec.external.license and spec.external.determinism


def test_unsupported_labels_explain_why_and_have_no_storage():
    unsupported = [s for s in REGISTRY if s.availability is Availability.UNSUPPORTED]
    assert unsupported
    for spec in unsupported:
        assert spec.unsupported_reason
        assert spec.table is None and not spec.columns


def test_human_annotation_labels_are_never_buildable():
    for spec in REGISTRY:
        if spec.source_kind is SourceKind.HUMAN_ANNOTATION:
            assert spec.availability is Availability.UNSUPPORTED


def test_higher_level_constructs_are_known_but_never_fabricated():
    for family in ("social_states", "addressee", "backchannel"):
        specs = [spec for spec in REGISTRY if spec.family == family]
        assert specs and all(s.availability is Availability.UNSUPPORTED for s in specs)
    for name in ("text.dialogue_act", "overlap.overlap_function", "video.gaze_proxy"):
        assert next(s for s in REGISTRY if s.name == name).extractor is None


def test_available_labels_never_claim_a_key_column():
    for spec in REGISTRY:
        if spec.table is not None:
            assert not set(spec.columns) & set(TABLE_KEYS[spec.table])


def test_a_physical_column_has_one_owner_per_extractor_table():
    owners: dict[tuple, str] = {}
    for spec in REGISTRY:
        if spec.row_filter is None and spec.table is not None:
            for column in spec.columns:
                key = (spec.extractor, spec.table, column)
                assert key not in owners, (spec.name, owners.get(key))
                owners[key] = spec.name


def test_the_document_has_versions_and_no_undocumented_field():
    document = json.loads(json.dumps(registry_document(REGISTRY)))
    assert document["registry_version"] == REGISTRY_VERSION
    assert document["label_schema_version"] == LABEL_SCHEMA_VERSION
    for entry in document["labels"]:
        assert set(entry) == DOCUMENTED_FIELDS


def test_validation_rejects_a_duplicate_name():
    with pytest.raises(RegistryError, match="duplicate"):
        validate((REGISTRY[0], REGISTRY[0]))


def test_validation_rejects_an_unknown_family():
    spec = replace(REGISTRY[0], name="mood.valence", family="mood")
    with pytest.raises(RegistryError, match="unknown family"):
        validate((spec,))


def test_validation_rejects_an_unsupported_label_without_reason():
    spec = next(s for s in REGISTRY if s.extractor is None)
    with pytest.raises(RegistryError, match="reason"):
        validate((replace(spec, unsupported_reason=None),))


def test_validation_rejects_an_external_label_without_tool():
    spec = next(s for s in REGISTRY if s.name == "prosody.ego_f0_hz")
    with pytest.raises(RegistryError, match="tool"):
        validate((replace(spec, external=None),))


def test_validation_rejects_a_table_the_level_cannot_use():
    spec = next(s for s in REGISTRY if s.name == "instantaneous.ego_speaking")
    with pytest.raises(RegistryError, match="table not allowed"):
        validate((replace(spec, table=Table.EVENTS),))


def test_support_is_computed_from_declared_facts_not_dataset_names():
    rows = support_matrix(REGISTRY, {"anything": {facts.SPEECH}})
    supported = {row.label for row in rows if row.supported}
    assert "timing.time_to_next_ego_onset" in supported
    assert "social_native.looking_at_wearer_subframes" not in supported
    lam = next(
        r for r in rows if r.label == "social_native.looking_at_wearer_subframes"
    )
    assert lam.missing_facts == (facts.FACE_TRACKS, facts.SOCIAL_LOOKING)


def test_unsupported_labels_are_supported_by_no_dataset():
    every_fact = facts.KNOWN_FACTS
    for spec in REGISTRY:
        if spec.extractor is None:
            assert not spec.supported_by(every_fact)


def test_counts_cover_every_source_kind():
    counts = label_counts()
    assert counts["labels"] == len(REGISTRY)
    assert set(counts["by_source_kind"]) == {str(kind) for kind in SourceKind}


def test_extractors_that_decode_media_are_exactly_audio_and_video():
    assert {e for e in Extractor if e.decodes_media} == {
        Extractor.AUDIO,
        Extractor.VIDEO,
    }


def test_media_extractor_labels_require_media_facts():
    for spec in REGISTRY:
        if spec.extractor is Extractor.AUDIO:
            assert facts.MEDIA_AUDIO in spec.requires
        if spec.extractor is Extractor.VIDEO:
            assert facts.MEDIA_VIDEO in spec.requires


def test_a_participant_axis_is_declared_by_the_shape():
    by = {spec.name: spec for spec in REGISTRY}
    assert by["instantaneous.speaker_activity_subframes"].participant_axis
    assert not by["instantaneous.ego_speaking_subframes"].participant_axis
    assert not by["turns.turns"].participant_axis


def test_label_spec_is_hashable_and_frozen():
    spec: LabelSpec = REGISTRY[0]
    with pytest.raises(AttributeError):
        spec.name = "x.y"  # type: ignore[misc]
    assert hash(spec)
