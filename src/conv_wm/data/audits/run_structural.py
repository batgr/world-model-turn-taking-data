"""Command-line runner for the phase A2 structural audit."""

import json
from pathlib import Path

import pandas as pd

from conv_wm.config import get_path, load_config
from conv_wm.data.audits.ego4d import audit_ego4d
from conv_wm.data.audits.egocom import audit_egocom
from conv_wm.data.schemas import EGO4D_SCHEMAS


def build_structural_report(
    video_info: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ego4d_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Build one report from already-loaded project tables."""
    egocom = audit_egocom(video_info, ground_truth)
    ego4d = audit_ego4d(ego4d_tables)
    return {
        "valid": bool(egocom["valid"] and ego4d["valid"]),
        "egocom": egocom,
        "ego4d": ego4d,
    }


def run_structural(config_path: str | Path | None = None) -> tuple[dict, Path]:
    """Load configured tables, audit them, and write the JSON report."""
    cfg = load_config(config_path)
    video_info = pd.read_csv(get_path(cfg, "egocom", "raw", "video_info"))
    ground_truth = pd.read_csv(get_path(cfg, "egocom", "raw", "ground_truth"))
    ego4d_tables = {
        name: pd.read_parquet(get_path(cfg, "ego4d", "interim", name))
        for name in EGO4D_SCHEMAS
    }

    report = build_structural_report(video_info, ground_truth, ego4d_tables)
    report_path = Path(cfg.paths.reports) / "structural" / "structural_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report, report_path


def _print_summary(report: dict, report_path: Path) -> None:
    for dataset_name in ("egocom", "ego4d"):
        dataset = report[dataset_name]
        print(dataset_name)
        for table_name, result in dataset["tables"].items():
            status = "PASS" if result["valid"] else "FAIL"
            print(f"  table {table_name}: {status} ({result['n_failures']} failures)")
        for relation_name, result in dataset["relations"].items():
            status = "PASS" if result["valid"] else "FAIL"
            print(
                f"  relation {relation_name}: {status} "
                f"({result['n_failures']} failures)"
            )
    print(f"overall: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"report: {report_path}")


def main() -> None:
    """Run A2 against configured data and print a concise summary."""
    report, report_path = run_structural()
    _print_summary(report, report_path)


if __name__ == "__main__":
    main()
