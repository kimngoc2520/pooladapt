"""Build SciFact's BM25+dense RRF candidate pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_beir_dataset
from src.baselines.fixed_prefix import select
from src.retrieval import BM25Retriever, DEFAULT_MODEL, DenseRetriever, fuse_ranked_lists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--queries", type=int, default=1, help="Number of queries to run; 0 runs all.")
    parser.add_argument("--prefix-budgets", type=int, nargs="+", default=[10, 20, 30, 50])
    args = parser.parse_args()

    corpus, queries, _ = load_beir_dataset(args.dataset_dir)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=args.model)
    query_items = list(queries.items())
    if args.queries:
        query_items = query_items[: args.queries]
    for query_id, query in tqdm(query_items, desc="Retrieval", unit="query"):
        pool = fuse_ranked_lists({"bm25": bm25.retrieve(query), "dense": dense.retrieve(query)})
        print(f"query_id={query_id} candidates={len(pool)} ids={[candidate['id'] for candidate in pool[:10]]}")
        print("prefix_sizes=" + repr({budget: len(select(pool, m=budget)) for budget in args.prefix_budgets}))


if __name__ == "__main__":
    main()
