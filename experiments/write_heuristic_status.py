"""Write the Phase 2 single-feature heuristic status artifact.

G2 failed under the locked selection criterion, so this deliberately writes
only the infeasibility configuration and never evaluates a heuristic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT = Path("results/phase2/07_heuristic/heuristic_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = {
        "heuristic_status": "infeasible_no_significant_feature",
        "g2_status": "FAILED",
        "g3_status": "NOT_EVALUATED",
        "g2_selection_criterion": {"abs_rho_gt": 0.2, "holm_p_lt": 0.05},
        "g2_reason": "No feature met the predefined Feature ↔ M* selection criterion.",
        "tau": 0.99,
        "development_query_count": 809,
        "source_analysis_files": [
            "results/phase2/03_analysis/feature_mstar_analysis_tau099.csv",
            "results/phase2/03_analysis/feature_mstar_audit_tau099.json",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
