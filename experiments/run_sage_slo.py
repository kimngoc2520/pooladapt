"""Development-only SAGE-inspired adaptive-K baseline; never reads TEST data."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.baselines.sage_slo import model_logits, select_budget, validate_feature_names
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k

FEATURES = ("top1_top2_rrf_margin", "top20_overlap", "rank_correlation_union", "score_entropy", "score_gini", "redundancy", "concentration", "sparse_dense_agreement", "sparse_dense_disagreement", "query_length", "num_numeric_tokens", "avg_idf", "max_idf")
FIELDS = ("query_id", "predicted_budget", "temperature", "budget_bias_lambda", "nDCG@10", "Recall@10", "MRR@10", "reranked_pairs", "compression_ratio", "reranking_latency")
SAGE_SLO_OUTPUT_DIR = Path("results/phase2/09_sage_slo")

def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def index(path: Path): return {str(row["query_id"]): row for row in read_csv(path)}
def read_ids(path: Path): return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def rows(pool, query, query_ids):
    out=[]
    for qid in query_ids:
        if qid not in pool or qid not in query: raise ValueError(f"missing existing feature row for {qid}")
        merged={**pool[qid], **query[qid]}
        out.append({name: float(merged[name]) if merged.get(name, "") != "" else np.nan for name in FEATURES})
    return out
def qrels(path: Path):
    out={}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"): out.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])]=int(row["score"])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-dir", type=Path, default=Path("results/phase2/01_candidate_logging")); p.add_argument("--mstar", type=Path, default=Path("results/phase2/02_oracle/mstar_labels_tau099.csv"))
    p.add_argument("--train-ids", type=Path, default=Path("data/scifact/splits/train_query_ids.txt")); p.add_argument("--validation-ids", type=Path, default=Path("data/scifact/splits/validation_query_ids.txt")); p.add_argument("--validation-qrels", type=Path, default=Path("data/scifact/qrels/validation.tsv")); p.add_argument("--validation-results", type=Path, default=Path("results/phase2/development_reranking/validation/reranking_results.json")); p.add_argument("--output", type=Path, default=SAGE_SLO_OUTPUT_DIR / "sage_slo_results.csv"); p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--budget-bias-lambda", type=float, default=0.0); p.add_argument("--seed", type=int, default=42)
    a=p.parse_args(); validate_feature_names(FEATURES)
    pool, query=index(a.feature_dir/"pool_features.csv"), index(a.feature_dir/"query_features.csv"); train, validation=read_ids(a.train_ids), read_ids(a.validation_ids)
    if set(train)&set(validation): raise ValueError("TRAIN and validation IDs overlap")
    labels={str(x["query_id"]):int(x["M_star"]) for x in read_csv(a.mstar)}
    if not set(train+validation)<=set(labels): raise ValueError("M* labels do not cover locked development IDs")
    train_rows, validation_rows=rows(pool,query,train),rows(pool,query,validation)
    model=Pipeline((("imputer",SimpleImputer(strategy="median")),("forest",RandomForestClassifier(n_estimators=100,max_depth=10,random_state=a.seed,class_weight="balanced"))))
    model.fit([[r[n] for n in FEATURES] for r in train_rows],[labels[x] for x in train])
    methods=json.loads(a.validation_results.read_text(encoding="utf-8"))["methods"]; validation_qrels=qrels(a.validation_qrels); output=[]
    for qid,row in zip(validation,validation_rows):
        budget=select_budget(model_logits(model,row,FEATURES),a.temperature,a.budget_bias_lambda)
        prediction=next(x for x in methods[f"fixed_prefix_{budget}"]["predictions"] if str(x["query_id"])==qid); ranked=[str(x) for x in prediction["document_ids"]]; rel=validation_qrels[qid]
        output.append({"query_id":qid,"predicted_budget":budget,"temperature":a.temperature,"budget_bias_lambda":a.budget_bias_lambda,"nDCG@10":ndcg_at_k(ranked,rel),"Recall@10":recall_at_k(ranked,rel),"MRR@10":mrr_at_k(ranked,rel),"reranked_pairs":budget,"compression_ratio":1-budget/100,"reranking_latency":prediction["latency_rerank_seconds"]})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8",newline="") as f: writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader(); writer.writerows(output)
    print(json.dumps({"output":str(a.output),"validation_queries":len(output),"average_predicted_budget":float(np.mean([r["predicted_budget"] for r in output])),"temperature":a.temperature,"budget_bias_lambda":a.budget_bias_lambda},indent=2))
if __name__=="__main__": main()
