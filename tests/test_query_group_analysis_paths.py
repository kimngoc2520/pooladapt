"""Path-resolution checks for the relocated Query-group Analysis scripts."""
from pathlib import Path

from improvement.improvement_1.query_group_analysis.scripts.common import OUT, ROOT


def test_query_group_analysis_output_root_is_inside_improvement_1() -> None:
    """Resolve repository inputs and QGA outputs from the shared helper."""
    repository_root = Path(__file__).resolve().parents[1]

    assert ROOT == repository_root
    assert OUT == repository_root / "improvement" / "improvement_1" / "query_group_analysis"
