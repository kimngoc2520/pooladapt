"""Run the development-only PoolAdapt adaptive-K policy on validation."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.pooladapt.selector import FROZEN_FEATURES, PoolAdaptSelector
from src.pooladapt.checkpoints import checkpoint_metadata, file_digest, load_checkpoint, metadata_path, save_checkpoint

FIELDS = ("query_id", "predicted_budget", "temperature", "budget_bias_lambda", "nDCG@10", "Recall@10", "MRR@10", "reranked_pairs", "compression_ratio", "reranking_latency")

def build_model(seed: int = 42) -> Pipeline:
    """Construct the unchanged existing PoolAdapt training pipeline."""
    return Pipeline((("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("classifier",LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))))

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def ids(path): return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
def index(path): return {str(row["query_id"]): row for row in read_csv(path)}
def feature_rows(pool, query, query_ids):
    result=[]
    for qid in query_ids:
        if qid not in pool or qid not in query: raise ValueError(f"missing frozen feature row for {qid}")
        merged={**pool[qid], **query[qid]}
        result.append({name: float(merged[name]) if merged.get(name, "") != "" else np.nan for name in FROZEN_FEATURES})
    return result
def load_qrels(path):
    result={}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"): result.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])]=int(row["score"])
    return result
def distribution(budgets):
    return {str(b): int(budgets.count(b)) for b in (10,20,30,50,100)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-dir", type=Path, default=Path("results/phase2/01_candidate_logging")); p.add_argument("--mstar", type=Path, default=Path("results/phase2/02_oracle/mstar_labels_tau099.csv")); p.add_argument("--train-ids", type=Path, default=Path("data/scifact/splits/train_query_ids.txt")); p.add_argument("--validation-ids", type=Path, default=Path("data/scifact/splits/validation_query_ids.txt")); p.add_argument("--validation-qrels", type=Path, default=Path("data/scifact/qrels/validation.tsv")); p.add_argument("--validation-results", type=Path, default=Path("results/phase2/development_reranking/validation/reranking_results.json")); p.add_argument("--output", type=Path, default=Path("results/phase2/10_pooladapt/pooladapt_results.csv")); p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--budget-bias-lambda", type=float, default=0.0); p.add_argument("--seed", type=int, default=42)
    checkpoints=p.add_mutually_exclusive_group()
    checkpoints.add_argument('--load-checkpoint', type=Path, help='Reuse a trusted fitted pipeline without fitting.')
    checkpoints.add_argument('--save-checkpoint', type=Path, help='Save the fitted TRAIN pipeline with reload verification.')
    a=p.parse_args()
    pool, query=index(a.feature_dir/"pool_features.csv"),index(a.feature_dir/"query_features.csv"); train, validation=ids(a.train_ids),ids(a.validation_ids)
    if set(train)&set(validation): raise ValueError("TRAIN and validation IDs overlap")
    labels={str(row["query_id"]):int(row["M_star"]) for row in read_csv(a.mstar)}
    if not set(train+validation)<=set(labels): raise ValueError("M* labels do not cover the locked development split")
    train_rows, validation_rows=feature_rows(pool,query,train),feature_rows(pool,query,validation)
    training_inputs=[[row[n] for n in FROZEN_FEATURES] for row in train_rows]
    if a.load_checkpoint:
        model=load_checkpoint(a.load_checkpoint, expected_name='PoolAdapt')
        metadata=json.loads(metadata_path(a.load_checkpoint).read_text(encoding='utf-8'))
        if metadata['training_query_ids'] != train or metadata['random_seed'] != a.seed:
            raise ValueError('checkpoint TRAIN population/order or seed differs from this run')
    else:
        model=build_model(seed=a.seed)
        model.fit(training_inputs,[labels[qid] for qid in train])
    if a.save_checkpoint:
        sources=(a.train_ids,a.mstar,a.feature_dir/'pool_features.csv',a.feature_dir/'query_features.csv')
        metadata=checkpoint_metadata(model,'PoolAdapt',train,{str(path):file_digest(path) for path in sources})
        save_checkpoint(model,a.save_checkpoint,metadata,training_inputs[:16],train[:16])
    selector=PoolAdaptSelector(model); methods=json.loads(a.validation_results.read_text(encoding="utf-8"))["methods"]; validation_qrels=load_qrels(a.validation_qrels); output=[]
    for qid,row in zip(validation,validation_rows):
        budget=selector.predict(row,temperature=a.temperature,budget_bias_lambda=a.budget_bias_lambda)
        prediction=next(x for x in methods[f"fixed_prefix_{budget}"]["predictions"] if str(x["query_id"])==qid); ranked=[str(x) for x in prediction["document_ids"]]; relevance=validation_qrels[qid]
        output.append({"query_id":qid,"predicted_budget":budget,"temperature":a.temperature,"budget_bias_lambda":a.budget_bias_lambda,"nDCG@10":ndcg_at_k(ranked,relevance),"Recall@10":recall_at_k(ranked,relevance),"MRR@10":mrr_at_k(ranked,relevance),"reranked_pairs":budget,"compression_ratio":1-budget/100,"reranking_latency":prediction["latency_rerank_seconds"]})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8",newline="") as f: writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader(); writer.writerows(output)
    budgets=[row["predicted_budget"] for row in output]
    print(json.dumps({"output":str(a.output),"split":"validation","training_split":"official_train_only","average_K":float(np.mean(budgets)),"min_K":min(budgets),"max_K":max(budgets),"budget_distribution":distribution(budgets),"budget_percentages":{k:v/len(budgets)*100 for k,v in distribution(budgets).items()},"temperature":a.temperature,"budget_bias_lambda":a.budget_bias_lambda,"calibration_status":"sanity_only_not_final_matched_budget"},indent=2))
if __name__=="__main__": main()
