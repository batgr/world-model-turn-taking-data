"""Selection semantics: exact names, ``family.*``, ``all`` and modality filters."""

from __future__ import annotations

import pytest

from conv_wm.data.labels.catalog import REGISTRY
from conv_wm.data.labels.registry import Availability, Modality
from conv_wm.data.labels.selection import (
    LabelSelection,
    LabelSelectionError,
    LabelUnavailableError,
    resolve,
)


def names(include, modalities=()):
    selection = LabelSelection(True, tuple(include), tuple(modalities))
    return set(resolve(selection, REGISTRY).names)


AVAILABLE = {s.name for s in REGISTRY if s.availability is Availability.AVAILABLE}


def test_disabled_selection_resolves_to_nothing():
    assert not resolve(LabelSelection(), REGISTRY)
    assert not resolve(LabelSelection(False, ("all",)), REGISTRY)


def test_the_default_mapping_is_disabled():
    assert LabelSelection.from_mapping(None) == LabelSelection()
    assert LabelSelection.from_mapping({}) == LabelSelection()
    assert LabelSelection.from_mapping({"enabled": False}).enabled is False


def test_all_selects_every_available_label_and_reports_the_others():
    resolved = resolve(LabelSelection.everything(), REGISTRY)
    assert set(resolved.names) == AVAILABLE
    assert set(resolved.skipped) == {s.name for s in REGISTRY} - AVAILABLE
    assert all(reason.startswith("unsupported") for reason in resolved.skipped.values())


def test_one_exact_label():
    assert names(["instantaneous.speaker_activity"]) == {
        "instantaneous.speaker_activity"
    }


def test_a_family_wildcard_selects_the_family_only():
    selected = names(["timing.*"])
    assert selected and all(name.startswith("timing.") for name in selected)
    assert selected == {s.name for s in REGISTRY if s.family == "timing"}


def test_entries_are_unioned_in_registry_order():
    selection = LabelSelection(
        True,
        ("events.*", "timing.time_to_next_ego_onset", "instantaneous.speaker_activity"),
    )
    resolved = resolve(selection, REGISTRY)
    order = [s.name for s in REGISTRY]
    assert list(resolved.names) == sorted(resolved.names, key=order.index)
    assert "timing.time_to_next_ego_onset" in resolved.names


def test_audio_modality_keeps_only_audio_only_labels():
    selected = names(["all"], ["audio"])
    specs = [s for s in REGISTRY if s.name in selected]
    assert specs and all(set(s.modalities) == {Modality.AUDIO} for s in specs)


def test_video_modality_keeps_only_video_only_labels():
    selected = names(["all"], ["video"])
    specs = [s for s in REGISTRY if s.name in selected]
    assert specs and all(set(s.modalities) == {Modality.VIDEO} for s in specs)
    assert "social_native.talking_to_wearer_subframes" not in selected


def test_audio_and_video_also_keep_multimodal_labels():
    selected = names(["all"], ["audio", "video"])
    assert "social_native.talking_to_wearer_subframes" in selected  # [audio, video]
    assert "instantaneous.speaker_activity" in selected
    assert "social_native.looking_at_wearer_subframes" in selected
    assert not any(name.startswith("text.") for name in selected)
    assert not any(name.startswith("metadata.") for name in selected)


def test_an_unknown_label_is_a_clear_error_with_a_suggestion():
    with pytest.raises(LabelSelectionError, match="Did you mean"):
        names(["instantaneous.speaker_activty"])


def test_an_unknown_family_is_an_error():
    with pytest.raises(LabelSelectionError, match="unknown label family"):
        names(["mood.*"])


def test_an_unknown_modality_is_an_error():
    with pytest.raises(LabelSelectionError, match="unknown modalities"):
        names(["all"], ["smell"])


def test_an_explicitly_named_unsupported_label_is_unavailable():
    with pytest.raises(LabelUnavailableError, match="unsupported"):
        names(["social_states.dominance"])


def test_a_wildcard_over_unsupported_labels_skips_them():
    resolved = resolve(LabelSelection(True, ("social_states.*",)), REGISTRY)
    assert not resolved.names
    assert set(resolved.skipped) == {
        s.name for s in REGISTRY if s.family == "social_states"
    }


def test_an_exact_label_outside_the_modalities_is_an_error():
    with pytest.raises(LabelSelectionError, match="requires modalities"):
        names(["text.tokens"], ["audio"])


def test_enabled_without_include_is_an_error():
    with pytest.raises(LabelSelectionError, match="include is empty"):
        resolve(LabelSelection(True, ()), REGISTRY)


def test_mapping_accepts_comma_separated_strings():
    selection = LabelSelection.from_mapping(
        {"enabled": True, "include": "timing.*,events.*", "modalities": "audio"}
    )
    assert selection.include == ("timing.*", "events.*")
    assert selection.modalities == ("audio",)


def test_mapping_rejects_unknown_keys():
    with pytest.raises(LabelSelectionError, match="unknown selection keys"):
        LabelSelection.from_mapping({"enabled": True, "families": ["x"]})
