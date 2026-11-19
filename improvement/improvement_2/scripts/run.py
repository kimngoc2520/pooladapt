"""Resumable development-only experiment execution.

The runner uses persisted prefix rankings only.  It never reads TEST inputs or
runs a cross encoder. The required output directory makes a new execution's
human-readable identity explicit.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, platform, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import percentileofscore
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, mrr_at_k
from improvement.improvement_2.scripts.protocol import (BUDGETS, align_probabilities,
    bisection_policy, closed_form_difference, hard_probability, holm_adjust, shifted_budget)

SEED, BOOTSTRAPS, PERMUTATIONS = 20261117, 2000, 10000
TARGETS = (11, 12, 14, 17, 20)
QUERY = ("query_length", "num_numeric_tokens", "avg_idf", "max_idf")
POOL = ("top1_top2_rrf_margin", "top20_overlap", "rank_correlation_union", "score_entropy", "score_gini", "redundancy", "concentration", "sparse_dense_disagreement")
CONFIGS = {"full": QUERY + POOL, "query_only": QUERY, "pool_only": POOL}


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else ["status"]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed); values = np.asarray(values, float)
    sample = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(sample,.025)), float(np.quantile(sample,.975))


def sign_flip(values: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed); values = np.asarray(values, float); observed = abs(values.mean())
    null = (rng.choice((-1,1), size=(PERMUTATIONS,len(values))) * values).mean(axis=1)
    return float((1 + (abs(null) >= observed).sum()) / (PERMUTATIONS + 1))


def strata(labels: np.ndarray) -> np.ndarray:
    return np.select([labels == 10, np.isin(labels, (20,30))], [0,1], default=2)


def make_model(c: float, seed: int) -> Pipeline:
    return Pipeline((("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=seed))))


def select_c(features: np.ndarray, labels: np.ndarray, seed: int, binary: bool = False) -> tuple[float, list[dict[str, Any]]]:
    target = (labels > 10).astype(int) if binary else labels
    split = StratifiedKFold(3, shuffle=True, random_state=seed)
    diagnostics: list[dict[str, Any]] = []
    for c in (.01,.1,1.,10.):
        scores = np.zeros(len(labels)) if binary else np.zeros((len(labels),5))
        for train, valid in split.split(features, strata(labels)):
            model = make_model(c, seed); model.fit(features[train], target[train])
            probability = model.predict_proba(features[valid])
            if binary:
                scores[valid] = probability[:, list(model.named_steps["classifier"].classes_).index(1)] if 1 in model.named_steps["classifier"].classes_ else 0
            else:
                scores[valid], _ = align_probabilities(model.named_steps["classifier"].classes_, probability)
        hard_score = scores if binary else hard_probability(scores)
        diagnostics.append({"C":c, "pooled_inner_hard_pr_auc":float(average_precision_score(labels>10,hard_score)),
                            "multiclass_log_loss":float(log_loss(labels,scores,labels=BUDGETS)) if not binary else ""})
    best = max(x["pooled_inner_hard_pr_auc"] for x in diagnostics)
    return min(x["C"] for x in diagnostics if np.isclose(x["pooled_inner_hard_pr_auc"],best)), diagnostics


def load_inputs() -> tuple[list[str], np.ndarray, dict[str,np.ndarray], np.ndarray, np.ndarray, dict[str,str]]:
    id_files = (ROOT/"data/scifact/splits/train_query_ids.txt", ROOT/"data/scifact/splits/validation_query_ids.txt")
    # This repository freezes TEST IDs in qrels rather than a separate ID file.
    test_ids = {row["query-id"] for row in read_csv(ROOT/"data/scifact/qrels/test.tsv", "\t")}
    ids = [x.strip() for path in id_files for x in path.read_text().splitlines() if x.strip()]
    if len(ids) != 809 or len(ids) != len(set(ids)) or test_ids.intersection(ids): raise ValueError("invalid development/TEST split")
    labels = {x["query_id"]:int(x["M_star"]) for x in read_csv(ROOT/"results/phase2/02_oracle/mstar_labels_tau099.csv")}
    pf = {x["query_id"]:x for x in read_csv(ROOT/"results/phase2/01_candidate_logging/pool_features.csv")}
    qf = {x["query_id"]:x for x in read_csv(ROOT/"results/phase2/01_candidate_logging/query_features.csv")}
    if set(ids)-labels.keys() or set(ids)-pf.keys() or set(ids)-qf.keys(): raise ValueError("input coverage failure")
    data = {name:np.array([[float(({**pf[q],**qf[q]})[f]) if ({**pf[q],**qf[q]})[f] else np.nan for f in cols] for q in ids]) for name,cols in CONFIGS.items()}
    qrels: dict[str,dict[str,int]] = {}
    for path in (ROOT/"data/scifact/qrels/train.tsv",ROOT/"data/scifact/qrels/validation.tsv"):
        for row in read_csv(path,"\t"): qrels.setdefault(row["query-id"],{})[row["corpus-id"]]=int(row["score"])
    metrics = np.zeros((len(ids),5,3)); latency = np.full((len(ids),5),np.nan); index={q:i for i,q in enumerate(ids)}
    for path in (ROOT/"results/phase2/development_reranking/train/reranking_results.json",ROOT/"results/phase2/development_reranking/validation/reranking_results.json"):
        methods=json.loads(path.read_text())["methods"]
        for j,k in enumerate(BUDGETS):
            for row in methods[f"fixed_prefix_{k}"]["predictions"]:
                i=index[str(row["query_id"])]; ranking=[str(x) for x in row["document_ids"]]
                metrics[i,j]=[ndcg_at_k(ranking,qrels[ids[i]]),recall_at_k(ranking,qrels[ids[i]]),mrr_at_k(ranking,qrels[ids[i]])]
                latency[i,j]=float(row.get("latency_rerank_seconds","nan"))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in (*id_files,ROOT/"results/phase2/02_oracle/mstar_labels_tau099.csv",ROOT/"results/phase2/01_candidate_logging/pool_features.csv",ROOT/"results/phase2/01_candidate_logging/query_features.csv")}
    return ids, np.array([labels[q] for q in ids]), data, metrics, latency, hashes


def stage_ct20(out: Path, ids: list[str], labels: np.ndarray, quality: np.ndarray) -> None:
    stage=out/"oracle_headroom"; oracle_index=np.searchsorted(BUDGETS,labels)
    paired=quality[np.arange(len(ids)),oracle_index,0]-quality[:,0,0]
    mean,low,high=bootstrap_mean(paired,SEED)
    rows=[{"query_id":q,"M_star":int(labels[i]),"fixed10_ndcg":quality[i,0,0],"oracle_ndcg":quality[i,oracle_index[i],0],"difference":paired[i]} for i,q in enumerate(ids)]
    atomic_csv(stage/"oracle_fixed10_query_level.csv",rows)
    atomic_csv(stage/"gate_a.csv",[{"estimate":mean,"ci95_low":low,"ci95_high":high,"delta_A":.005,"decision":"above" if low>.005 else "below" if high<.005 else "inconclusive","oracle_average_K":float(labels.mean())}])
    envelope=[]
    for j,k in enumerate(BUDGETS):
        gap=quality[:,j,0]-quality[:,-1,0]; envelope.append({"K":k,"mean_ndcg":quality[:,j,0].mean(),"prefix_exceeds_full_count":int((gap>0).sum()),"mean_signed_gap_to_full":gap.mean()})
    atomic_csv(stage/"full_prefix_envelope.csv",envelope)
    # Cost-matched oracle policy: enumerate mu breakpoints through a robust grid, then policy-level mix adjacent costs.
    mus=np.linspace(-1,1,40001); policies=[]
    for mu in mus:
        choice=BUDGETS[np.argmax(quality[:,:,0]-mu*BUDGETS,axis=1)]; cost=choice.mean()
        if not policies or cost != policies[-1][1]: policies.append((choice,cost,mu))
    records=[]
    for target in TARGETS:
        lowpol=min((p for p in policies if p[1]>=target), key=lambda p:p[1], default=None); highpol=max((p for p in policies if p[1]<=target), key=lambda p:p[1], default=None)
        if lowpol is None or highpol is None: records.append({"target_average_K":target,"feasible":False}); continue
        if np.isclose(lowpol[1],highpol[1]): weight=1.; expected=lowpol[1]; selected=lowpol[0]
        else: weight=(target-highpol[1])/(lowpol[1]-highpol[1]); expected=weight*lowpol[1]+(1-weight)*highpol[1]; selected=lowpol[0]
        oracle=np.where(np.arange(len(ids))>=0, weight*quality[np.arange(len(ids)),np.searchsorted(BUDGETS,lowpol[0]),0]+(1-weight)*quality[np.arange(len(ids)),np.searchsorted(BUDGETS,highpol[0]),0],0)
        null=quality[:,0,0]; gain=oracle-null; records.append({"target_average_K":target,"feasible":abs(expected-target)<=.1,"mix_probability_policy_A":weight,"policy_A_average_K":lowpol[1],"policy_B_average_K":highpol[1],"expected_average_K":expected,"oracle_gain_vs_fixed10":gain.mean()})
    atomic_csv(stage/"cost_matched_oracle.csv",records)
    # Sensitivity / MDE artifacts; policy simulation is reported separately from Gate C.
    sensitivity=[]
    for p in (.01,.025,.05,.1):
        expected=10+90*p; gain=p*(quality[:,np.searchsorted(BUDGETS,labels),0]-quality[:,0,0]).mean()
        sensitivity.append({"oracle_recovery_probability":p,"expected_average_K":expected,"expected_gain_vs_fixed10":gain})
    atomic_csv(stage/"oracle_recovery_sensitivity.csv",sensitivity)


def run_cv(out: Path, ids: list[str], labels: np.ndarray, feature_data: dict[str,np.ndarray]) -> tuple[dict[str,tuple[np.ndarray,np.ndarray]], list[dict[str,Any]]]:
    aggregate={}; fold_rows=[]
    for name,x in feature_data.items():
        pred: dict[int,list[np.ndarray]]=defaultdict(list); masks: dict[int,list[np.ndarray]]=defaultdict(list)
        for repeat in range(10):
            for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=repeat).split(x,strata(labels))):
                chosen,diagnostics=select_c(x[train],labels[train],repeat*100+fold)
                model=make_model(chosen,repeat*100+fold); model.fit(x[train],labels[train])
                probability,available=align_probabilities(model.named_steps["classifier"].classes_,model.predict_proba(x[test]))
                for local,i in enumerate(test):
                    pred[i].append(probability[local]); masks[i].append(available)
                    fold_rows.append({"configuration":name,"repeat":repeat,"fold":fold,"query_id":ids[i],"selected_C":chosen,"class_available":"|".join(map(str,available.astype(int))),**{f"inner_C_{d['C']}_pr_auc":d["pooled_inner_hard_pr_auc"] for d in diagnostics}})
        probability=np.array([np.mean(pred[i],axis=0) for i in range(len(ids))])
        # A class remains available after aggregation only if it was available in every contributing OOF model.
        available=np.array([np.all(masks[i],axis=0) for i in range(len(ids))])
        aggregate[name]=(probability,available)
    atomic_csv(out/"feature_ablation"/"fold_predictions.csv",fold_rows)
    return aggregate,fold_rows


def rank_swap_p(y: np.ndarray, one: np.ndarray, two: np.ndarray, seed: int) -> float:
    one=np.array([percentileofscore(one,x,kind="mean")/100 for x in one]); two=np.array([percentileofscore(two,x,kind="mean")/100 for x in two])
    observed=average_precision_score(y,one)-average_precision_score(y,two); rng=np.random.default_rng(seed); exceed=0
    for _ in range(PERMUTATIONS):
        swap=rng.integers(0,2,len(y)).astype(bool); a=np.where(swap,two,one); b=np.where(swap,one,two)
        exceed += abs(average_precision_score(y,a)-average_precision_score(y,b)) >= abs(observed)
    return (1+exceed)/(PERMUTATIONS+1)


def stage_ct21(out: Path, ids: list[str], labels: np.ndarray, quality: np.ndarray, aggregate: dict[str,tuple[np.ndarray,np.ndarray]]) -> dict[str,dict[int,np.ndarray]]:
    stage=out/"feature_ablation"; hard=labels>10; prediction_rows=[]; metrics=[]; policies: dict[str,dict[int,np.ndarray]]={}
    for name,(prob,available) in aggregate.items():
        score=hard_probability(prob); pred=BUDGETS[np.argmax(prob,axis=1)]; p,r,f,s=precision_recall_fscore_support(hard,pred>10,zero_division=0)
        ap=average_precision_score(hard,score); _,lo,hi=bootstrap_mean(np.array([average_precision_score(hard[np.random.default_rng(SEED+i).choice(len(hard),len(hard),replace=True)],score[np.random.default_rng(SEED+i).choice(len(hard),len(hard),replace=True)]) for i in range(BOOTSTRAPS)]),SEED) if False else (0,0,0)
        # Direct bootstrap PR-AUC over paired query resamples.
        rng=np.random.default_rng(SEED); aps=np.array([average_precision_score(hard[ix],score[ix]) for ix in rng.integers(0,len(hard),(BOOTSTRAPS,len(hard)))])
        metrics.append({"configuration":name,"hard_pr_auc":ap,"ci95_low":np.quantile(aps,.025),"ci95_high":np.quantile(aps,.975),"hard_prevalence":hard.mean(),"hard_precision_argmax":p[1],"hard_recall_argmax":r[1],"hard_f1_argmax":f[1],"three_group_macro_f1":f1_score(strata(labels),strata(pred),average="macro",zero_division=0),"three_group_balanced_accuracy":balanced_accuracy_score(strata(labels),strata(pred)),"five_class_macro_f1":f1_score(labels,pred,labels=BUDGETS,average="macro",zero_division=0)})
        policies[name]={}
        for target in TARGETS:
            # Per-query all-available mask: lambda cannot revive a class absent in any repeat for that query.
            choice,lam,cost,feasible=bisection_policy(prob,available,target)
            policies[name][target]=choice
            for i,q in enumerate(ids): prediction_rows.append({"configuration":name,"query_id":q,"M_star":labels[i],"hard_true":int(hard[i]),"hard_score":score[i],"argmax_K":pred[i],"target_average_K":target,"lambda":lam,"policy_K":choice[i],"class_available":"|".join(map(str,available[i].astype(int))),**{f"p_{k}":prob[i,j] for j,k in enumerate(BUDGETS)}})
    atomic_csv(stage/"oof_predictions.csv",prediction_rows); atomic_csv(stage/"prediction_metrics.csv",metrics)
    # Gate B paired primary / secondary score-swap tests and Holm family.
    scores={name:hard_probability(value[0]) for name,value in aggregate.items()}; pairs=[("full","query_only","primary"),("full","pool_only","secondary"),("pool_only","query_only","secondary")]
    comparison=[]; secondary=[]
    for i,(a,b,family) in enumerate(pairs):
        diff=scores[a]-scores[b]; _,lo,hi=bootstrap_mean(diff,SEED+i); pvalue=rank_swap_p(hard,scores[a],scores[b],SEED+i)
        row={"contrast":f"{a}_minus_{b}","family":family,"score_difference_mean":diff.mean(),"ci95_low":lo,"ci95_high":hi,"raw_p":pvalue}; comparison.append(row); (secondary.append(row) if family=="secondary" else None)
    adjusted=holm_adjust([x["raw_p"] for x in secondary])
    for row,value in zip(secondary,adjusted): row["holm_adjusted_p"]=value
    atomic_csv(stage/"gate_b_comparisons.csv",comparison)
    frontier=[]
    for name,by_target in policies.items():
        for target,chosen in by_target.items():
            diff=closed_form_difference(chosen,quality[:,:,0]); mean,lo,hi=bootstrap_mean(diff,SEED+target)
            frontier.append({"configuration":name,"target_average_K":target,"realized_average_K":chosen.mean(),"feasible":abs(chosen.mean()-target)<=.1,"gain_vs_closed_form_null":mean,"ci95_low":lo,"ci95_high":hi,"signflip_p":sign_flip(diff,SEED+target)})
    atomic_csv(stage/"frontier.csv",frontier)
    common=[t for t in TARGETS if all(abs(policies[n][t].mean()-t)<=.1 for n in CONFIGS)]
    gate=[]
    endpoint={}
    for name in CONFIGS:
        # Recompute all nulls inside every query bootstrap resample, across common levels.
        rng=np.random.default_rng(SEED); values=[]
        original=[]
        for target in common: original.append(closed_form_difference(policies[name][target],quality[:,:,0]))
        point=np.mean([x.mean() for x in original])
        for ix in rng.integers(0,len(ids),(BOOTSTRAPS,len(ids))): values.append(np.mean([closed_form_difference(policies[name][t],quality[:,:,0],ix).mean() for t in common]))
        endpoint[name]=np.array(values)
        gate.append({"configuration":name,"common_targets":"|".join(map(str,common)),"primary_gain":point,"ci95_low":np.quantile(values,.025),"ci95_high":np.quantile(values,.975),"delta_C":.0025,"decision":"above" if np.quantile(values,.025)>.0025 else "below" if np.quantile(values,.975)<.0025 else "inconclusive"})
    atomic_csv(stage/"gate_c_endpoints.csv",gate)
    fc=[]; secondary=[]
    for i,(a,b,family) in enumerate(pairs):
        diffs=np.mean([closed_form_difference(policies[a][t],quality[:,:,0])-closed_form_difference(policies[b][t],quality[:,:,0]) for t in common],axis=0)
        mean,lo,hi=bootstrap_mean(diffs,SEED+i); row={"contrast":f"{a}_minus_{b}","family":family,"frontier_difference":mean,"ci95_low":lo,"ci95_high":hi,"raw_p":sign_flip(diffs,SEED+i)};fc.append(row); (secondary.append(row) if family=="secondary" else None)
    for row,value in zip(secondary,holm_adjust([x["raw_p"] for x in secondary])): row["holm_adjusted_p"]=value
    atomic_csv(stage/"gate_c_comparisons.csv",fc)
    return policies


def stage_ct22(out: Path, ids: list[str], labels: np.ndarray, quality: np.ndarray, latency: np.ndarray, aggregate: dict[str,tuple[np.ndarray,np.ndarray]], policies: dict[str,dict[int,np.ndarray]]) -> None:
    stage=out/"controller_diagnosis"; prob,_=aggregate["full"]; hard=labels>10; argmax=BUDGETS[np.argmax(prob,axis=1)]; score=hard_probability(prob)
    records=[]
    for i,q in enumerate(ids):
        j=np.searchsorted(BUDGETS,argmax[i]); o=np.searchsorted(BUDGETS,labels[i])
        records.append({"query_id":q,"M_star":labels[i],"argmax_K":argmax[i],"hard_true":int(hard[i]),"hard_score":score[i],"hard_predicted":int(argmax[i]>10),"signed_K_error":argmax[i]-labels[i],"absolute_K_error":abs(argmax[i]-labels[i]),"ordinal_index_error":j-o,"underallocated":int(argmax[i]<labels[i]),"overallocated":int(argmax[i]>labels[i]),"nDCG10":quality[i,j,0],"Recall10":quality[i,j,1],"MRR10":quality[i,j,2],"nDCG_loss_vs_oracle":quality[i,o,0]-quality[i,j,0],"additional_pairs_vs_oracle":argmax[i]-labels[i],"latency":latency[i,j]})
    atomic_csv(stage/"allocation_errors.csv",records)
    cm=confusion_matrix(labels,argmax,labels=BUDGETS); atomic_csv(stage/"five_class_confusion.csv",[{"actual_K":k,**{f"pred_{z}":int(cm[i,j]) for j,z in enumerate(BUDGETS)}} for i,k in enumerate(BUDGETS)])
    p,r,f,s=precision_recall_fscore_support(hard,argmax>10,zero_division=0); atomic_csv(stage/"hard_detection.csv",[{"hard_pr_auc":average_precision_score(hard,score),"hard_precision":p[1],"hard_recall":r[1],"hard_f1":f[1],"prevalence":hard.mean()}])
    comparison=[]
    for target,choice in policies["full"].items():
        j=np.searchsorted(BUDGETS,choice); comparison.append({"policy":"full_lambda_shift","target_average_K":target,"actual_average_K":choice.mean(),"mean_ndcg":quality[np.arange(len(ids)),j,0].mean(),"mean_recall":quality[np.arange(len(ids)),j,1].mean(),"mean_mrr":quality[np.arange(len(ids)),j,2].mean(),"mean_latency":np.nanmean(latency[np.arange(len(ids)),j]),"p50_latency":np.nanquantile(latency[np.arange(len(ids)),j],.5),"p95_latency":np.nanquantile(latency[np.arange(len(ids)),j],.95)})
    for j,k in enumerate(BUDGETS): comparison.append({"policy":f"fixed_prefix_{k}","target_average_K":k,"actual_average_K":k,"mean_ndcg":quality[:,j,0].mean(),"mean_recall":quality[:,j,1].mean(),"mean_mrr":quality[:,j,2].mean(),"mean_latency":np.nanmean(latency[:,j]),"p50_latency":np.nanquantile(latency[:,j],.5),"p95_latency":np.nanquantile(latency[:,j],.95)})
    atomic_csv(stage/"quality_cost_comparison.csv",comparison)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True,help="New, human-readable execution directory"); parser.add_argument("--protocol",type=Path,required=True); args=parser.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    ids,labels,features,quality,latency,hashes=load_inputs(); manifest={"run_id":out.name,"status":"running","protocol_version":"v8.2","protocol_path":str(args.protocol),"protocol_sha256":digest(args.protocol),"test_used":False,"input_hashes":hashes,"feature_configurations":{x:list(y) for x,y in CONFIGS.items()},"outer_cv":"RepeatedStratifiedKFold 5x10 seeds 0..9","inner_cv":"StratifiedKFold 3","C_grid":[.01,.1,1,10],"seed":SEED,"bootstrap_draws":BOOTSTRAPS,"permutations":PERMUTATIONS,"platform":platform.platform(),"python":sys.version}; atomic_json(out/"manifest.json",manifest)
    stage_ct20(out,ids,labels,quality); aggregate,_=run_cv(out,ids,labels,features); policies=stage_ct21(out,ids,labels,quality,aggregate); stage_ct22(out,ids,labels,quality,latency,aggregate,policies)
    manifest["status"]="completed"; atomic_json(out/"manifest.json",manifest); print(out)

if __name__ == "__main__": main()
