"""Run official TRAIN+VALIDATION feature--M* associations after Oracle labels exist."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.correlation import TAUS, analyze, load_feature_sets, multicollinearity, read_rows

FIELDS = ["feature_name", "feature_level", "aggregation_method", "tau", "spearman_rho", "p_value", "p_holm", "abs_rho", "selected"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("results/phase2/01_candidate_logging"))
    parser.add_argument("--mstar-dir", type=Path, default=Path("results/phase2/02_oracle"), help="Directory containing official train+validation M* CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2/03_analysis"))
    args = parser.parse_args()
    if "phase2_test_descriptive" in {part.lower() for part in args.mstar_dir.parts}:
        raise ValueError("TEST-descriptive artifacts must never be used for official Feature--M* analysis")
    label_paths = {tau: args.mstar_dir / f"mstar_labels_tau{tau}.csv" for tau in TAUS}
    missing = [str(path) for path in label_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Official TRAIN+VALIDATION M* artifacts are required; missing: " + ", ".join(missing))
    features = load_feature_sets(args.features_dir / "candidate_features.csv", args.features_dir / "pool_features.csv", args.features_dir / "query_features.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tau, label_path in label_paths.items():
        rows = analyze(features, read_rows(label_path), tau)
        with (args.output_dir / f"feature_mstar_analysis_tau{tau}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in FIELDS} for row in rows)
        audit_rows = [{"feature_name": row["feature_name"], "feature_level": row["feature_level"], "number_of_rows_used": row["number_of_rows_used"], "number_of_unique_queries": row["number_of_unique_queries"]} for row in rows]
        with (args.output_dir / f"feature_mstar_audit_tau{tau}.json").open("w", encoding="utf-8") as handle:
            json.dump(audit_rows, handle, indent=2)
            handle.write("\n")
    with (args.output_dir / "multicollinearity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_level", "feature_a", "feature_b", "pearson_r"])
        writer.writeheader()
        writer.writerows(multicollinearity(features))
    print("Wrote official TRAIN+VALIDATION associations. Results are associations, not causal effects.")


if __name__ == "__main__":
    main()
