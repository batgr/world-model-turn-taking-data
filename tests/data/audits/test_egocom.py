from conv_wm.data.audits.structural import audit_dataset_structure
from conv_wm.data.datasets.egocom import EGOCOM


def _tables(video_info, ground_truth):
    return {"video_info": video_info, "ground_truth_transcriptions": ground_truth}


def test_egocom_structure_accepts_nullable_raw_transcription_fields(
    valid_egocom_tables,
):
    result = audit_dataset_structure(EGOCOM, _tables(*valid_egocom_tables)).to_dict()

    assert result["valid"] is True
    assert result["tables"]["video_info"]["valid"] is True
    assert result["tables"]["ground_truth_transcriptions"]["valid"] is True
    relation = result["relations"]["ground_truth_conversation_id_in_video_info"]
    assert relation["orphan_rows"] == 0


def test_egocom_structure_reports_contract_and_relation_failures(valid_egocom_tables):
    video_info, ground_truth = valid_egocom_tables
    video_info.loc[1, "video_id"] = 1
    video_info.loc[1, "num_speakers"] = 0
    video_info.loc[1, "train"] = True
    ground_truth.loc[1, "conversation_id"] = "missing"

    result = audit_dataset_structure(
        EGOCOM, _tables(video_info, ground_truth)
    ).to_dict()

    assert result["valid"] is False
    assert result["tables"]["video_info"]["n_failures"] >= 3
    relation = result["relations"]["ground_truth_conversation_id_in_video_info"]
    assert relation["orphan_rows"] == 1
