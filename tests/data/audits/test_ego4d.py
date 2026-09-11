from conv_wm.data.audits.ego4d import audit_ego4d


def test_audit_ego4d_accepts_valid_tables(valid_ego4d_tables):
    report = audit_ego4d(valid_ego4d_tables)

    assert report["valid"] is True
    assert all(result["valid"] for result in report["tables"].values())
    assert all(result["valid"] for result in report["relations"].values())


def test_audit_ego4d_reports_orphan_person_reference(valid_ego4d_tables):
    valid_ego4d_tables["voice_segments_clean"].loc[0, "person_id"] = "missing"

    report = audit_ego4d(valid_ego4d_tables)

    relation = report["relations"]["voice_segments_person_in_persons"]
    assert relation == {"valid": False, "n_failures": 1, "orphan_rows": 1}
    assert report["valid"] is False
