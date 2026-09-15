from conv_wm.data.audits.structural import audit_dataset_structure
from conv_wm.data.datasets.ego4d import EGO4D


def test_ego4d_structure_accepts_valid_tables(valid_ego4d_tables):
    result = audit_dataset_structure(EGO4D, valid_ego4d_tables)

    assert result.valid is True
    assert all(table.valid for table in result.tables)
    assert all(relation.valid for relation in result.relations)


def test_ego4d_structure_reports_orphan_person_reference(valid_ego4d_tables):
    valid_ego4d_tables["voice_segments_clean"].loc[0, "person_id"] = "missing"

    result = audit_dataset_structure(EGO4D, valid_ego4d_tables)

    relation = result.to_dict()["relations"]["voice_segments_person_in_persons"]
    assert relation == {"valid": False, "n_failures": 1, "orphan_rows": 1}
    assert result.valid is False
