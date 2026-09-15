from conv_wm.data.audits.annotations import audit_dataset_annotations
from conv_wm.data.audits.structural import audit_dataset_structure
from conv_wm.data.datasets.egocom import EGOCOM
from conv_wm.data.datasets.egocom_cleaning import clean_ground_truth, clean_video_info


def _tables(video_info, ground_truth):
    return {
        "video_info": video_info,
        "ground_truth_transcriptions": ground_truth,
        "video_info_clean": clean_video_info(video_info).table,
        "ground_truth_clean": clean_ground_truth(ground_truth).table,
    }


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
    tables = _tables(video_info, ground_truth)
    video_info.loc[1, "video_id"] = 1
    video_info.loc[1, "num_speakers"] = 0
    video_info.loc[1, "train"] = True
    ground_truth.loc[1, "conversation_id"] = "missing"

    result = audit_dataset_structure(EGOCOM, tables).to_dict()

    assert result["valid"] is False
    assert result["tables"]["video_info"]["n_failures"] >= 3
    relation = result["relations"]["ground_truth_conversation_id_in_video_info"]
    assert relation["orphan_rows"] == 1


def test_ground_truth_participant_without_own_pov_is_not_an_integrity_failure(
    valid_egocom_tables,
):
    video_info, ground_truth = valid_egocom_tables
    ground_truth.loc[0, "speaker_id"] = 3
    tables = {
        "video_info": clean_video_info(video_info).table,
        "ground_truth_transcriptions": clean_ground_truth(ground_truth).table,
    }

    result = audit_dataset_annotations(EGOCOM, tables, media=None)

    ground_truth_report = next(
        source
        for source in result.sources
        if source.name == "ground_truth_transcriptions"
    )
    assert ground_truth_report.valid
    assert all(
        "participant" not in anomaly.constraint
        for anomaly in ground_truth_report.anomalies
    )
