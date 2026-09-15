from conv_wm.data.audits.structural import (
    audit_dataset_structure,
    build_structural_report,
)
from conv_wm.data.datasets.ego4d import EGO4D
from conv_wm.data.datasets.egocom import EGOCOM


def test_structural_report_combines_datasets(valid_egocom_tables, valid_ego4d_tables):
    video_info, ground_truth = valid_egocom_tables
    results = [
        audit_dataset_structure(
            EGOCOM,
            {"video_info": video_info, "ground_truth_transcriptions": ground_truth},
        ),
        audit_dataset_structure(EGO4D, valid_ego4d_tables),
    ]

    report = build_structural_report(results)

    assert report["valid"] is True
    assert set(report["datasets"]) == {"egocom", "ego4d"}
    assert report["datasets"]["ego4d"]["valid"] is True
