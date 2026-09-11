import pandas as pd

from conv_wm.data.audits.pandera import audit_dataframe
from conv_wm.data.schemas import (
    EGOCOM_GROUND_TRUTH_SCHEMA,
    EGOCOM_VIDEO_INFO_SCHEMA,
)
from conv_wm.data.validation import check_foreign_key


def audit_egocom(
    video_info: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, object]:
    """Run structural checks on EgoCom tables."""

    tables = {
        "video_info": audit_dataframe(video_info, EGOCOM_VIDEO_INFO_SCHEMA),
        "ground_truth_transcriptions": audit_dataframe(
            ground_truth,
            EGOCOM_GROUND_TRUTH_SCHEMA,
        ),
    }
    orphan_rows = check_foreign_key(
        ground_truth,
        ["conversation_id"],
        video_info,
        ["conversation_id"],
    )
    relations = {
        "ground_truth_conversation_id_in_video_info": {
            "valid": orphan_rows == 0,
            "n_failures": orphan_rows,
            "orphan_conversation_rows": orphan_rows,
        }
    }

    return {
        "valid": all(result["valid"] for result in tables.values())
        and all(result["valid"] for result in relations.values()),
        "tables": tables,
        "relations": relations,
    }
