from conv_wm.data.audits.egocom import audit_egocom


def test_audit_egocom_accepts_nullable_raw_transcription_fields(
    valid_egocom_tables,
):
    report = audit_egocom(*valid_egocom_tables)

    assert report["valid"] is True
    assert report["tables"]["video_info"]["valid"] is True
    assert report["tables"]["ground_truth_transcriptions"]["valid"] is True
    relation = report["relations"]["ground_truth_conversation_id_in_video_info"]
    assert relation["orphan_conversation_rows"] == 0


def test_audit_egocom_reports_contract_and_relation_failures(valid_egocom_tables):
    video_info, ground_truth = valid_egocom_tables
    video_info.loc[1, "video_id"] = 1
    video_info.loc[1, "num_speakers"] = 0
    video_info.loc[1, "train"] = True
    ground_truth.loc[1, "conversation_id"] = "missing"

    report = audit_egocom(video_info, ground_truth)

    assert report["valid"] is False
    assert report["tables"]["video_info"]["n_failures"] >= 3
    relation = report["relations"]["ground_truth_conversation_id_in_video_info"]
    assert relation["orphan_conversation_rows"] == 1
