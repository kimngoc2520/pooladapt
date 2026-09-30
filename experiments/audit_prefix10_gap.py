from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import ndcg_at_k


RESULTS = [
    Path("results/phase2/development_reranking/train/reranking_results.json"),
    Path("results/phase2/development_reranking/validation/reranking_results.json"),
]

QRELS = [
    Path("data/scifact/qrels/train_split.tsv"),
    Path("data/scifact/qrels/validation.tsv"),
]


def load_qrels(path):
    qrels = {}

    with path.open(encoding="utf-8") as f:
        next(f, None)

        for line in f:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    return qrels


def main():
    qrels = {}

    for path in QRELS:
        qrels.update(load_qrels(path))

    gaps = []

    for result_path in RESULTS:
        with result_path.open(encoding="utf-8") as f:
            artifact = json.load(f)

        full_predictions = {
            str(x["query_id"]): x["document_ids"]
            for x in artifact["methods"]["full_rerank"]["predictions"]
        }

        prefix_predictions = {
            str(x["query_id"]): x["document_ids"]
            for x in artifact["methods"]["fixed_prefix_10"]["predictions"]
        }

        for query_id in full_predictions:
            full_ndcg = ndcg_at_k(
                full_predictions[query_id],
                qrels[query_id],
                k=10,
            )

            prefix_ndcg = ndcg_at_k(
                prefix_predictions[query_id],
                qrels[query_id],
                k=10,
            )

            gaps.append({
                "query_id": query_id,
                "full_ndcg": full_ndcg,
                "prefix10_ndcg": prefix_ndcg,
                "gap": full_ndcg - prefix_ndcg,
            })

    values = [x["gap"] for x in gaps]

    print("=" * 60)
    print("PREFIX-10 QUALITY GAP AUDIT")
    print("=" * 60)

    print(f"Queries: {len(values)}")
    print(f"Mean gap:   {statistics.mean(values):.6f}")
    print(f"Median gap: {statistics.median(values):.6f}")
    print(f"Max gap:    {max(values):.6f}")
    print(f"Min gap:    {min(values):.6f}")

    for threshold in [0.001, 0.01, 0.02, 0.05, 0.10]:
        count = sum(x > threshold for x in values)
        print(
            f"gap > {threshold:.3f}: "
            f"{count}/{len(values)} "
            f"({count / len(values) * 100:.2f}%)"
        )

    sorted_gaps = sorted(gaps, key=lambda x: x["gap"], reverse=True)

    print("\nTop 20 largest quality gaps:")
    print("query_id | Full | Prefix10 | Gap")

    for row in sorted_gaps[:20]:
        print(
            f"{row['query_id']} | "
            f"{row['full_ndcg']:.4f} | "
            f"{row['prefix10_ndcg']:.4f} | "
            f"{row['gap']:.4f}"
        )

    output = Path("results/phase2/04_prefix10_gap_audit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "full_ndcg", "prefix10_ndcg", "gap"],
        )
        writer.writeheader()
        writer.writerows(gaps)

    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()
