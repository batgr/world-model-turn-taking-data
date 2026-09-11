from conv_wm.data.audits.run_structural import build_structural_report


def test_build_structural_report_combines_both_datasets(
    valid_egocom_tables,
    valid_ego4d_tables,
):
    report = build_structural_report(*valid_egocom_tables, valid_ego4d_tables)

    assert report["valid"] is True
    assert report["egocom"]["valid"] is True
    assert report["ego4d"]["valid"] is True
