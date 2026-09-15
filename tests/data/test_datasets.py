import pytest

from conv_wm.data import datasets
from conv_wm.data.datasets.spec import DatasetSpec


def test_builtin_datasets_are_registered():
    assert datasets.names() == ["ego4d", "egocom"]
    assert datasets.get("ego4d").audio.known_boundary_grid is not None
    assert datasets.get("egocom").audio.known_boundary_grid is None


def test_unregistered_dataset_gets_an_empty_spec():
    spec = datasets.get("someday")

    assert not datasets.is_registered("someday")
    assert spec.name == "someday"
    assert spec.audio.known_boundary_grid is None


def test_register_refuses_silent_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        datasets.register(DatasetSpec(name="ego4d"))


def test_spec_name_must_be_lower_case():
    with pytest.raises(ValueError):
        DatasetSpec(name="Ego4D")
