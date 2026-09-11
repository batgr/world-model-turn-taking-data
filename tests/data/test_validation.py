import pandas as pd
import pytest

from conv_wm.data.validation import check_foreign_key


def test_check_foreign_key_counts_missing_child_rows():
    parent = pd.DataFrame({"conversation_id": ["a", "b"]})
    child = pd.DataFrame({"conversation_id": ["a", "missing", "missing"]})

    result = check_foreign_key(
        child,
        ["conversation_id"],
        parent,
        ["conversation_id"],
    )

    assert result == 2


def test_check_foreign_key_supports_composite_keys():
    parent = pd.DataFrame({"conversation_id": ["a", "a"], "speaker_id": [1, 2]})
    child = pd.DataFrame({"conversation_id": ["a", "a"], "speaker_id": [1, 3]})

    result = check_foreign_key(
        child,
        ["conversation_id", "speaker_id"],
        parent,
        ["conversation_id", "speaker_id"],
    )

    assert result == 1


def test_check_foreign_key_rejects_mismatched_key_lengths():
    parent = pd.DataFrame({"conversation_id": ["a"], "speaker_id": [1]})
    child = pd.DataFrame({"conversation_id": ["a"]})

    with pytest.raises(ValueError, match="must have the same length"):
        check_foreign_key(
            child,
            ["conversation_id"],
            parent,
            ["conversation_id", "speaker_id"],
        )
