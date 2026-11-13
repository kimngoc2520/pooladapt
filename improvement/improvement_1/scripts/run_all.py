#!/usr/bin/env python3
"""Reproducible downstream analysis of the frozen PoolAdapt TEST handoff.

The script reads the handoff CSV and audit records, writes only beneath
improvement/improvement_1, and never changes Phase 2 source artifacts.
Random executions are averaged within query before query-level analyses.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, random, statistics, subprocess, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "improvement" / "improvement_1"
INPUT = OUT / "matched_budget" / "query_level_evaluation.csv"
SEED = 20261112
BOOTSTRAPS = 10_000
PERMUTATIONS = 100_000
MATCH_TOLERANCE_PCT = 5.0
HIGH_LATENCY_RULE = "Tukey upper fence: latency > Q3 + 1.5*IQR within method/operating point"
METHODS = ["PoolAdapt", "Fixed Prefix", "SAGE-SLO", "PACE-EF", "Random", "Full Rerank"]
METRICS = ["ndcg_at_10", "recall_at_10", "mrr_at_10"]
METRIC_LABELS = {"ndcg_at_10":"nDCG@10", "recall_at_10":"Recall@10", "mrr_at_10":"MRR@10"}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_rows(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]|None=None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for row in rows: w.writerow({k:row.get(k,"") for k in fields})

def num(r:dict[str,Any],key:str)->float: return float(r[key])
def mean(xs:list[float])->float: return float(statistics.fmean(xs)) if xs else math.nan
def pct(xs:list[float],p:float)->float: return float(np.percentile(xs,p,method="linear")) if xs else math.nan
def fmt(x:Any,d=6)->str:
    if x is None or (isinstance(x,float) and not math.isfinite(x)): return ""
    if isinstance(x,float): return f"{x:.{d}f}"
    return str(x)

def validate(rows:list[dict[str,str]])->dict[str,Any]:
    required=["query_id","query_text","method","operating_point","baseline_method","seed","selected_k","controller_features","actual_reranked_pairs","execution_id","execution_manifest_sha256","reranking_execution_count","ndcg_at_10","recall_at_10","mrr_at_10","reranking_latency_ms","latency_measurement_scope","execution_environment_reference","query_group"]
    errors=[]
    if not rows: errors.append("Input CSV contains no records")
    if list(rows[0])!=required: errors.append(f"Schema mismatch: observed {list(rows[0]) if rows else []}")
    qids=sorted({r["query_id"] for r in rows}); keys=set(); strata=defaultdict(list); missing=defaultdict(int); metric_ranges={k:[math.inf,-math.inf] for k in METRICS}
    for i,r in enumerate(rows,2):
        key=(r["query_id"],r["method"],r["operating_point"],r["seed"])
        if key in keys: errors.append(f"Duplicate composite key at CSV line {i}: {key}")
        keys.add(key); strata[(r["method"],r["operating_point"],r["seed"])].append(r["query_id"])
        for k,v in r.items():
            if v=="": missing[k]+=1
        try:
            k=float(r["selected_k"]); pairs=float(r["actual_reranked_pairs"]); lat=float(r["reranking_latency_ms"])
            if k<0 or pairs<0 or not math.isfinite(k+pairs): errors.append(f"Invalid pair count line {i}")
            if abs(k-pairs)>1e-9: errors.append(f"selected_k differs from actual pairs at line {i}")
            if lat<0 or not math.isfinite(lat): errors.append(f"Invalid latency at line {i}")
            for m in METRICS:
                v=float(r[m]); metric_ranges[m]=[min(metric_ranges[m][0],v),max(metric_ranges[m][1],v)]
                if not math.isfinite(v) or not 0<=v<=1: errors.append(f"Invalid {m} at line {i}")
        except (ValueError,KeyError) as e: errors.append(f"Numeric parse error line {i}: {e}")
    for s,ids in strata.items():
        if len(ids)!=300 or set(ids)!=set(qids): errors.append(f"Incomplete stratum {s}: {len(ids)} rows, {len(set(ids))} unique queries")
    expected={(m,f"B{b}") for m in METHODS[:-1] for b in (10,20,30,50)}|{("Full Rerank","B100")}
    seen={(m,b) for m,b,_ in strata}
    if expected!=seen: errors.append(f"Method/operating point coverage mismatch; missing={expected-seen}, unexpected={seen-expected}")
    for (m,b,s),ids in strata.items():
        if m=="Random" and s not in {"0","1","2","3","4"}: errors.append(f"Unexpected Random seed {s}")
        if m!="Random" and s!="": errors.append(f"Unexpected seed for non-Random {m} {b}: {s}")
    if not all(r["query_text"] for r in rows): errors.append("Missing query text")
    if not all(r["execution_manifest_sha256"] for r in rows): errors.append("Missing execution manifest linkage")
    qrels=ROOT/"data/scifact/qrels/test.tsv"
    qrel_ids=set(); qrel_positive_rows=0
    if qrels.exists():
        with qrels.open(encoding="utf-8-sig") as f:
            next(f, None)  # SciFact handoff is query-id, corpus-id, score.
            for line in f:
                parts=line.strip().split()
                if len(parts)>=3 and float(parts[2])>0: qrel_ids.add(parts[0]); qrel_positive_rows+=1
    else: errors.append("Qrels file unavailable for positive-judgment cross-check")
    if not qrel_ids: errors.append("No positive qrels query IDs parsed from qrels input")
    qrel_coverage={q:sum(1 for r in rows if r["query_id"]==q) for q in qrel_ids}
    if len(qids)!=300: errors.append(f"Expected 300 unique query IDs; found {len(qids)}")
    qrel_missing=sorted(set(qids)-qrel_ids)
    if qrel_missing: errors.append(f"{len(qrel_missing)} TEST queries lack positive qrels")
    return {"errors":errors,"rows":len(rows),"queries":len(qids),"unique_composite_keys":len(keys),"strata":len(strata),"methods":sorted({r["method"] for r in rows}),"operating_points":sorted({r["operating_point"] for r in rows}),"random_seeds":sorted({r["seed"] for r in rows if r["method"]=="Random"}),"missing_by_column":dict(missing),"metric_ranges":metric_ranges,"positive_qrels_queries":len(qrel_ids),"positive_qrels_judgments":qrel_positive_rows,"positive_qrels_coverage_missing":qrel_missing,"manifest_hashes":sorted({r["execution_manifest_sha256"] for r in rows}),"stratum_sizes":{f"{m}/{b}/seed={s or 'NA'}":len(ids) for (m,b,s),ids in sorted(strata.items())}}

def aggregate(rows:list[dict[str,str]])->list[dict[str,Any]]:
    """Return one row/query/method/point; Random seed values averaged within query."""
    groups=defaultdict(list)
    for r in rows: groups[(r["query_id"],r["method"],r["operating_point"])].append(r)
    out=[]
    avg_fields=["selected_k","actual_reranked_pairs","reranking_latency_ms","reranking_execution_count",*METRICS]
    for (qid,m,b),rs in sorted(groups.items(),key=lambda z:(int(z[0][0]),z[0][1],z[0][2])):
        d={k:rs[0][k] for k in ("query_id","query_text","method","operating_point","baseline_method","execution_manifest_sha256","latency_measurement_scope","execution_environment_reference","query_group")}
        d["seed_count"]=len(rs); d["seed_policy"]="mean within query" if m=="Random" else "single execution"
        for k in avg_fields:d[k]=mean([float(r[k]) for r in rs])
        d["seed_values"]=";".join(r["seed"] for r in rs)
        out.append(d)
    return out

def summarize(agg:list[dict[str,Any]])->list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in agg:groups[(r["method"],r["operating_point"])].append(r)
    full=mean([num(r,"actual_reranked_pairs") for r in groups[("Full Rerank","B100")]])
    output=[]
    for (m,b),rs in sorted(groups.items(),key=lambda z:(METHODS.index(z[0][0]),int(z[0][1][1:]))):
        pairs=[num(r,"actual_reranked_pairs") for r in rs]; lat=[num(r,"reranking_latency_ms") for r in rs]
        d={"method":m,"operating_point":b,"n_queries":len(rs),"average_k":mean([num(r,"selected_k") for r in rs]),"average_actual_pairs":mean(pairs),"compression_ratio":mean(pairs)/full,"pair_cost_reduction":1-mean(pairs)/full,"mean_reranking_latency_ms":mean(lat),"p50_reranking_latency_ms":pct(lat,50),"p95_reranking_latency_ms":pct(lat,95)}
        for met in METRICS:d[met]=mean([num(r,met) for r in rs])
        output.append(d)
    return output

def random_seed_sensitivity(raw:list[dict[str,str]])->list[dict[str,Any]]:
    """Summarize between-seed spread descriptively; seeds are not query replicates."""
    groups=defaultdict(list)
    for r in raw:
        if r["method"]=="Random":groups[(r["operating_point"],r["seed"])].append(r)
    per_seed=[]
    for (op,seed),rs in sorted(groups.items(),key=lambda z:(int(z[0][0][1:]),int(z[0][1]))):
        d={"operating_point":op,"seed":seed,"n_queries":len(rs)}
        for m in METRICS:d[m]=mean([num(r,m) for r in rs])
        d["average_actual_pairs"]=mean([num(r,"actual_reranked_pairs") for r in rs]);d["mean_reranking_latency_ms"]=mean([num(r,"reranking_latency_ms") for r in rs])
        per_seed.append(d)
    result=[]
    for op in ("B10","B20","B30","B50"):
        subset=[r for r in per_seed if r["operating_point"]==op]
        d={"method":"Random","operating_point":op,"seed_count":len(subset),"queries_per_seed":300,"query_population":"300 TEST queries per seed; seeds summarized descriptively"}
        for key in [*METRICS,"average_actual_pairs","mean_reranking_latency_ms"]:
            vals=[num(r,key) for r in subset];d[f"{key}_seed_mean"]=mean(vals);d[f"{key}_seed_sd"]=statistics.stdev(vals) if len(vals)>1 else math.nan;d[f"{key}_seed_min"]=min(vals);d[f"{key}_seed_max"]=max(vals)
        result.append(d)
    return result

def compare_source_aggregates(summaries:list[dict[str,Any]],raw:list[dict[str,str]])->dict[str,Any]:
    """Reproduce the frozen 21-row summary without adopting its legacy cost label."""
    path=ROOT/"results/phase2/evaluation/final_gpu/phase2_summary.json"
    if not path.exists():return {"available":False,"matched_rows":0,"mismatches":["Source aggregate summary unavailable"]}
    source=json.loads(path.read_text(encoding="utf-8"))["final_test_results"]
    source_map={(r["method"],f"B{int(r['target_budget'])}"):r for r in source}
    fields=[("ndcg_at_10","nDCG@10",1.0),("recall_at_10","Recall@10",1.0),("mrr_at_10","MRR@10",1.0),("average_actual_pairs","avg_reranked_pairs",1.0),("mean_reranking_latency_ms","mean_reranking_latency",1000.0),("p50_reranking_latency_ms","p50_reranking_latency",1000.0),("p95_reranking_latency_ms","p95_reranking_latency",1000.0)]
    mismatches=[]; compared=0
    if len(source_map)!=21:mismatches.append(f"Expected 21 source aggregate rows; found {len(source_map)}")
    for cur in summaries:
        key=(cur["method"],cur["operating_point"]); ref=source_map.get(key)
        if ref is None:mismatches.append(f"Missing source aggregate row {key}");continue
        for cf,sf,scale in fields:
            # The frozen summary's Random percentiles are means of per-seed
            # percentiles. That legacy statistic is not the percentile of the
            # 300 within-query seed means used downstream, so leave these eight
            # source fields un-compared rather than pool seeds as independent
            # queries or imply they validate the downstream percentile.
            if cur["method"]=="Random" and cf in ("p50_reranking_latency_ms","p95_reranking_latency_ms"):
                continue
            else:
                delta=abs(float(cur[cf])-float(ref[sf])*scale)
            compared+=1
            if delta>1e-10:mismatches.append(f"Aggregate mismatch {key} {cf}: delta={delta:.3g}")
    return {"available":True,"source_summary_path":"results/phase2/evaluation/final_gpu/phase2_summary.json","source_rows":len(source_map),"compared_values":compared,"not_compared_values":8,"not_compared_note":"Historical Random P50/P95 source fields average per-seed percentiles. Improvement 1 designates percentiles of 300 within-query five-seed means as primary; see latency_cost/random_latency_percentile_sensitivity.csv for primary, per-seed, pooled, and historical estimates.","tolerance":1e-10,"matched_rows":len(summaries) if not mismatches else 0,"mismatches":mismatches}

def paired_stats(agg:list[dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    idx={(r["query_id"],r["method"],r["operating_point"]):r for r in agg}; raw=[]
    for b in (10,20,50):
        op=f"B{b}"
        for met in METRICS:
            ids=sorted(set(q for q,m,o in idx if m=="PoolAdapt" and o==op)&set(q for q,m,o in idx if m=="Fixed Prefix" and o==op),key=int)
            a=np.array([num(idx[(q,"PoolAdapt",op)],met) for q in ids]); c=np.array([num(idx[(q,"Fixed Prefix",op)],met) for q in ids]); diff=a-c; n=len(diff)
            rng=np.random.default_rng(SEED+100*b+METRICS.index(met)); boot=np.mean(rng.choice(diff,size=(BOOTSTRAPS,n),replace=True),axis=1)
            prng=np.random.default_rng(SEED+200*b+METRICS.index(met)); signs=prng.choice(np.array([-1.,1.]),size=(PERMUTATIONS,n)); perm=np.abs(np.mean(signs*diff,axis=1)); obs=abs(float(np.mean(diff))); pperm=(1+int(np.count_nonzero(perm>=obs)))/(PERMUTATIONS+1)
            try: wp=float(stats.wilcoxon(diff,zero_method="wilcox",alternative="two-sided",method="approx").pvalue)
            except (ValueError,ZeroDivisionError):wp=math.nan
            sd=float(np.std(diff,ddof=1)); dz=float(np.mean(diff)/sd) if sd>0 else math.nan
            raw.append({"comparison":f"PoolAdapt {op} vs Fixed Prefix {op}","operating_point":op,"metric":METRIC_LABELS[met],"n_queries":n,"pooladapt_mean":mean(a.tolist()),"fixed_prefix_mean":mean(c.tolist()),"mean_paired_difference":float(np.mean(diff)),"median_paired_difference":float(np.median(diff)),"bootstrap_95_ci_low":float(np.percentile(boot,2.5)),"bootstrap_95_ci_high":float(np.percentile(boot,97.5)),"permutation_p_raw":pperm,"wilcoxon_p_supplementary":wp,"cohen_dz":dz,"paired_difference_sd":sd,"zero_difference_count":int(np.sum(diff==0)),"bootstrap_replicates":BOOTSTRAPS,"permutation_replicates":PERMUTATIONS,"random_seed":SEED,"wilcoxon_zero_policy":"zero differences excluded (zero_method=wilcox)"})
    # Holm correction separately for each metric family (three budgets).
    for metric in {r["metric"] for r in raw}:
        family=[r for r in raw if r["metric"]==metric]; order=sorted(family,key=lambda r:r["permutation_p_raw"]); running=0.0; m=len(order)
        for rank,r in enumerate(order):
            adj=min(1.0,(m-rank)*r["permutation_p_raw"]); running=max(running,adj); r["permutation_p_holm_adjusted"]=running
    summary=[]
    for r in raw:
        summary.append({"operating_point":r["operating_point"],"metric":r["metric"],"n_queries":r["n_queries"],"mean_paired_difference":r["mean_paired_difference"],"median_paired_difference":r["median_paired_difference"],"bootstrap_95_ci_low":r["bootstrap_95_ci_low"],"bootstrap_95_ci_high":r["bootstrap_95_ci_high"],"permutation_p_raw":r["permutation_p_raw"],"permutation_p_holm_adjusted":r["permutation_p_holm_adjusted"],"cohen_dz":r["cohen_dz"]})
    return raw,summary

def pareto(summaries:list[dict[str,Any]])->list[dict[str,Any]]:
    out=[]
    for scope,xfield in [("quality_vs_actual_pairs","average_actual_pairs"),("quality_vs_reranking_latency","mean_reranking_latency_ms")]:
        pts=list(summaries)
        for p in pts:
            x=num(p,xfield); y=num(p,"ndcg_at_10")
            dominated=any((num(q,xfield)<=x and num(q,"ndcg_at_10")>=y and (num(q,xfield)<x or num(q,"ndcg_at_10")>y)) for q in pts if q is not p)
            out.append({**p,"pareto_objective":scope,"optimization_subset":"all methods and operating points, including Full Rerank B100","objective_x":xfield,"objective_x_value":x,"objective_y":"ndcg_at_10","objective_y_value":y,"pareto_efficient":not dominated,"tie_policy":"identical points remain non-dominated unless strictly dominated by another point"})
    return out

def spearman(x:list[float],y:list[float])->float:
    if len(set(x))<2 or len(set(y))<2:return math.nan
    return float(stats.spearmanr(x,y).statistic)

def latency(agg:list[dict[str,Any]])->list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in agg:groups[(r["method"],r["operating_point"])].append(r)
    result=[]
    for (m,b),rs in sorted(groups.items(),key=lambda z:(METHODS.index(z[0][0]),int(z[0][1][1:]))):
        lat=[num(r,"reranking_latency_ms") for r in rs]; q1=pct(lat,25); q3=pct(lat,75); fence=q3+1.5*(q3-q1); hi=[r for r in rs if num(r,"reranking_latency_ms")>fence]
        row={"method":m,"operating_point":b,"n_queries":len(rs),"mean_ms":mean(lat),"p50_ms":pct(lat,50),"p75_ms":q3,"p90_ms":pct(lat,90),"p95_ms":pct(lat,95),"p99_ms":pct(lat,99),"std_ms":statistics.stdev(lat) if len(lat)>1 else math.nan,"iqr_ms":q3-q1,"p95_minus_p50_ms":pct(lat,95)-pct(lat,50),"tukey_upper_fence_ms":fence,"high_latency_query_count":len(hi),"high_latency_query_fraction":len(hi)/len(rs),"high_latency_rule":HIGH_LATENCY_RULE,"scope":"reranking stage: pair preparation + Cross-Encoder inference + sorting; excludes retrieval, selection, model load and warm-up"}
        for fld,label in [("selected_k","selected_k"),("actual_reranked_pairs","actual_reranked_pairs"),*[(m,m) for m in METRICS]]:
            x=[num(r,fld) for r in rs]; row[f"spearman_latency_vs_{label}"]=spearman(lat,x)
        pair_x=np.asarray([num(r,"actual_reranked_pairs") for r in rs],dtype=float); latency_y=np.asarray(lat,dtype=float)
        if np.var(pair_x)>0 and np.var(latency_y)>0:
            slope,intercept=np.polyfit(pair_x,latency_y,1); residual=latency_y-(intercept+slope*pair_x)
            row["latency_pairs_linear_r2"]=float(1-np.sum(residual**2)/np.sum((latency_y-np.mean(latency_y))**2));row["latency_pairs_residual_std_ms"]=float(np.std(residual,ddof=1));row["latency_pairs_linear_slope_ms_per_pair"]=float(slope)
        else:
            row["latency_pairs_linear_r2"]=math.nan;row["latency_pairs_residual_std_ms"]=statistics.stdev(lat) if len(lat)>1 else math.nan;row["latency_pairs_linear_slope_ms_per_pair"]=math.nan
        if hi:
            other=[r for r in rs if r not in hi]; row["high_latency_mean_ms"]=mean([num(r,"reranking_latency_ms") for r in hi]);row["remaining_mean_ms"]=mean([num(r,"reranking_latency_ms") for r in other]); row["high_latency_mean_pairs"]=mean([num(r,"actual_reranked_pairs") for r in hi]);row["remaining_mean_pairs"]=mean([num(r,"actual_reranked_pairs") for r in other])
        result.append(row)
    return result

def failures(agg:list[dict[str,Any]])->list[dict[str,Any]]:
    idx={(r["query_id"],r["method"],r["operating_point"]):r for r in agg}; cases=[]
    # Categories and thresholds are declared up front; all qualifying records retained.
    for b in (10,20,50):
        op=f"B{b}"
        for q in sorted({q for q,m,o in idx if m=="PoolAdapt" and o==op},key=int):
            p=idx[(q,"PoolAdapt",op)]; f=idx[(q,"Fixed Prefix",op)]; full=idx[(q,"Full Rerank","B100")]
            dp=num(p,"ndcg_at_10")-num(f,"ndcg_at_10"); excess=num(p,"actual_reranked_pairs")-num(f,"actual_reranked_pairs")
            common={"query_id":q,"query_text":p["query_text"],"operating_point":op,"selected_k":fmt(num(p,"selected_k")),"actual_reranked_pairs":fmt(num(p,"actual_reranked_pairs")),"PoolAdapt_nDCG@10":fmt(num(p,"ndcg_at_10")),"PoolAdapt_Recall@10":fmt(num(p,"recall_at_10")),"PoolAdapt_MRR@10":fmt(num(p,"mrr_at_10")),"Full_Rerank_nDCG@10":fmt(num(full,"ndcg_at_10")),"Full_Rerank_Recall@10":fmt(num(full,"recall_at_10")),"Full_Rerank_MRR@10":fmt(num(full,"mrr_at_10")),"Fixed_Prefix_reference":op,"Fixed_Prefix_actual_pairs":fmt(num(f,"actual_reranked_pairs")),"Fixed_Prefix_nDCG@10":fmt(num(f,"ndcg_at_10")),"Fixed_Prefix_Recall@10":fmt(num(f,"recall_at_10")),"Fixed_Prefix_MRR@10":fmt(num(f,"mrr_at_10")),"controller_features":"unavailable","retrieval_signals":"unavailable in TEST handoff"}
            if num(p,"actual_reranked_pairs")<num(f,"actual_reranked_pairs") and dp<=-0.20:
                cases.append({**common,"failure_category":"under-allocation candidate","selection_criterion":"PoolAdapt actual pairs < same-label Fixed Prefix and nDCG@10 difference <= -0.20","criterion_value":fmt(dp),"failure_rationale":f"Observed nDCG@10 was {abs(dp):.3f} lower while PoolAdapt reranked fewer pairs; comparison with full rerank is retained for context.","proposed_improvement":"Inspect candidate-pool decision signals after validated TEST features become available.","evidence_limitations":"Descriptive per-query observation; no controller features or causal evidence."})
            if excess>=max(2.0,0.10*num(f,"actual_reranked_pairs")) and dp<=0:
                cases.append({**common,"failure_category":"over-allocation candidate","selection_criterion":"PoolAdapt uses >=10% and >=2 more actual pairs than Fixed Prefix, with no nDCG@10 gain","criterion_value":fmt(excess),"failure_rationale":f"PoolAdapt spent {excess:.2f} additional pairs with nDCG@10 difference {dp:.3f}.","proposed_improvement":"Review selector thresholds and calibration against the matched prefix reference.","evidence_limitations":"Observed comparison only; no monetary/GPU costs or mechanism attribution."})
    # Include up to three documented Tukey high-latency observations per required
    # PoolAdapt tail stratum, ranked by measured latency. This is descriptive.
    for b in (20,30,50):
        op=f"B{b}"; pool=[r for r in agg if r["method"]=="PoolAdapt" and r["operating_point"]==op]
        vals=[num(r,"reranking_latency_ms") for r in pool]; q1=pct(vals,25); q3=pct(vals,75); fence=q3+1.5*(q3-q1)
        outliers=sorted([r for r in pool if num(r,"reranking_latency_ms")>fence],key=lambda r:-num(r,"reranking_latency_ms"))[:3]
        for rank,p in enumerate(outliers,1):
            q=p["query_id"]; f=idx[(q,"Fixed Prefix",op)]; full=idx[(q,"Full Rerank","B100")]
            cases.append({"query_id":q,"query_text":p["query_text"],"operating_point":op,"selected_k":fmt(num(p,"selected_k")),"actual_reranked_pairs":fmt(num(p,"actual_reranked_pairs")),"PoolAdapt_nDCG@10":fmt(num(p,"ndcg_at_10")),"PoolAdapt_Recall@10":fmt(num(p,"recall_at_10")),"PoolAdapt_MRR@10":fmt(num(p,"mrr_at_10")),"Full_Rerank_nDCG@10":fmt(num(full,"ndcg_at_10")),"Full_Rerank_Recall@10":fmt(num(full,"recall_at_10")),"Full_Rerank_MRR@10":fmt(num(full,"mrr_at_10")),"Fixed_Prefix_reference":op,"Fixed_Prefix_actual_pairs":fmt(num(f,"actual_reranked_pairs")),"Fixed_Prefix_nDCG@10":fmt(num(f,"ndcg_at_10")),"Fixed_Prefix_Recall@10":fmt(num(f,"recall_at_10")),"Fixed_Prefix_MRR@10":fmt(num(f,"mrr_at_10")),"controller_features":"unavailable","retrieval_signals":"unavailable in TEST handoff","failure_category":"high-latency case","selection_criterion":f"Tukey outlier > Q3 + 1.5*IQR in PoolAdapt {op}; top {rank} of highest latency outliers","criterion_value":fmt(num(p,"reranking_latency_ms")),"failure_rationale":f"Measured reranking latency {num(p,'reranking_latency_ms'):.1f} ms exceeded the stratum upper fence {fence:.1f} ms; actual pairs={num(p,'actual_reranked_pairs'):.0f}; Fixed Prefix latency={num(f,'reranking_latency_ms'):.1f} ms.","proposed_improvement":"Inspect per-pair execution timing and validated runtime logs before proposing a mechanism.","evidence_limitations":"Tail observation only; no evidence attributing latency to hardware contention or another cause."})
    # Deterministic severity ordering for downstream representative plots/review.
    cases.sort(key=lambda r:(r["failure_category"],-abs(float(r["criterion_value"])),int(r["query_id"]),r["operating_point"]))
    return cases

def svg_base(title:str, xlab:str, ylab:str, body:str, width=1000,height=640)->str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Arial,sans-serif;fill:#172033}} .small{{font-size:13px}} .axis{{stroke:#586174;stroke-width:1.5}} .grid{{stroke:#e5e9f0;stroke-width:1}} .title{{font-size:24px;font-weight:bold}} .label{{font-size:16px}} </style><text x="{width/2}" y="34" text-anchor="middle" class="title">{title}</text>{body}<text x="{width/2}" y="{height-16}" text-anchor="middle" class="label">{xlab}</text><text transform="translate(24 {height/2}) rotate(-90)" text-anchor="middle" class="label">{ylab}</text></svg>'''

def render_svg(svg:str,out:Path)->None:
    out.parent.mkdir(parents=True,exist_ok=True); temp=out.with_suffix(".svg");temp.write_text(svg,encoding="utf-8")
    edge=Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.exists(): raise RuntimeError("Microsoft Edge is required to render reproducible SVG figures to PNG")
    profile=OUT/".edge_profile"; profile.mkdir(exist_ok=True)
    uri=temp.resolve().as_uri()
    subprocess.run([str(edge),"--headless=new","--disable-gpu","--no-sandbox",f"--user-data-dir={profile}","--hide-scrollbars","--window-size=1000,640",f"--screenshot={out}",uri],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
    temp.unlink(missing_ok=True)

def figures(summaries:list[dict[str,Any]],agg:list[dict[str,Any]],latrows:list[dict[str,Any]],pareto_rows:list[dict[str,Any]],cases:list[dict[str,Any]])->None:
    # SVG figures are rendered by local headless Edge; figure data are built from the CSV-producing objects.
    colors={"PoolAdapt":"#2563eb","Fixed Prefix":"#dc2626","SAGE-SLO":"#059669","PACE-EF":"#7c3aed","Random":"#d97706","Full Rerank":"#111827"}
    def axes(xmax=100,ymin=0,ymax=1):
        L,T,R,B=90,70,950,560; body=[]
        for i in range(6):
            y=T+(B-T)*i/5; body.append(f'<line class="grid" x1="{L}" y1="{y}" x2="{R}" y2="{y}"/><text class="small" x="{L-10}" y="{y+4}" text-anchor="end">{ymax-(ymax-ymin)*i/5:.2f}</text>')
        body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>'])
        return body,(lambda x:L+(R-L)*x/xmax),(lambda y:B-(B-T)*(y-ymin)/(ymax-ymin))
    # Matched quality (PoolAdapt and Fixed Prefix) as three aligned panels in one figure.
    body=[]; L,R=90,950; xx=[10,20,50]; ys=(["ndcg_at_10","recall_at_10","mrr_at_10"])
    for j,met in enumerate(ys):
        top=80+j*155; bot=205+j*155
        body.append(f'<text class="small" x="55" y="{top+60}" transform="rotate(-90 55 {top+60})">{METRIC_LABELS[met]}</text>')
        body.append(f'<line class="grid" x1="{L}" y1="{bot}" x2="{R}" y2="{bot}"/><line class="axis" x1="{L}" y1="{top}" x2="{L}" y2="{bot}"/><line class="axis" x1="{L}" y1="{bot}" x2="{R}" y2="{bot}"/>')
        for tick in (0.0,0.5,1.0):
            ty=bot-(bot-top)*tick
            body.append(f'<line class="grid" x1="{L}" y1="{ty}" x2="{R}" y2="{ty}"/><text class="small" x="{L-8}" y="{ty+4}" text-anchor="end">{tick:.1f}</text>')
        for m in ("PoolAdapt","Fixed Prefix"):
            points=[]
            for b in xx:
                r=next(z for z in summaries if z["method"]==m and z["operating_point"]==f"B{b}"); x=L+(R-L)*(b-10)/40; y=bot-(bot-top)*num(r,met); points.append(f"{x},{y}")
            body.append(f'<polyline fill="none" stroke="{colors[m]}" stroke-width="3" points="{" ".join(points)}"/>')
            for pt in points:
                x,y=map(float,pt.split(','));body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[m]}"/>')
        for b in xx: body.append(f'<text class="small" x="{L+(R-L)*(b-10)/40}" y="{bot+20}" text-anchor="middle">B{b}</text>')
    body.append('<line x1="690" y1="55" x2="720" y2="55" stroke="#2563eb" stroke-width="3"/><text class="small" x="728" y="60">PoolAdapt</text><line x1="820" y1="55" x2="850" y2="55" stroke="#dc2626" stroke-width="3"/><text class="small" x="858" y="60">Fixed Prefix</text>')
    render_svg(svg_base("Matched-budget quality by nominal operating point","Operating point (B30 is supplementary; outside ±5% pair match)","Metric score",''.join(body),1000,640),OUT/"figures/matched_budget_quality_comparison.png")
    # Latency comparison means plus p95 for selected key operating points.
    picks=[("PoolAdapt",20),("PoolAdapt",30),("PoolAdapt",50),("SAGE-SLO",20),("SAGE-SLO",30),("Fixed Prefix",20),("Fixed Prefix",30),("Fixed Prefix",50)]
    lr=[next(z for z in latrows if z["method"]==m and z["operating_point"]==f"B{b}") for m,b in picks]; maxy=max(z["p95_ms"] for z in lr)*1.15
    body=[]; L,T,R,B=90,70,950,555; bw=28; gap=(R-L)/len(picks)
    tickstep=max(25,math.ceil(maxy/4/25)*25)
    for tick in range(0,int(maxy)+1,tickstep):
        ty=B-(B-T)*tick/maxy;body.append(f'<line class="grid" x1="{L}" y1="{ty}" x2="{R}" y2="{ty}"/><text class="small" x="{L-8}" y="{ty+4}" text-anchor="end">{tick}</text>')
    for i,z in enumerate(lr):
        x=L+gap*(i+.5); ymean=B-(B-T)*z["mean_ms"]/maxy; yp=B-(B-T)*z["p95_ms"]/maxy
        body.append(f'<rect x="{x-bw/2}" y="{ymean}" width="{bw}" height="{B-ymean}" fill="{colors[z["method"]]}" opacity=".75"/><line x1="{x}" y1="{yp}" x2="{x}" y2="{ymean}" stroke="#111827" stroke-width="3"/><text class="small" x="{x}" y="{B+18}" text-anchor="middle">{z["method"].replace(" ","-")} B{z["operating_point"][1:]}</text>')
    body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>',f'<text class="small" x="{L}" y="{T-12}">Bars: mean; whisker to P95</text>'])
    render_svg(svg_base("Reranking latency comparison","Method and operating point","Reranking latency (ms)",''.join(body),1000,640),OUT/"figures/latency_comparison.png")
    # Latency distributions: overlaid frequency polyline per required strata.
    keys=[("PoolAdapt",20),("PoolAdapt",30),("PoolAdapt",50),("SAGE-SLO",20),("SAGE-SLO",30)]; data={}
    series_colors={("PoolAdapt",20):"#2563eb",("PoolAdapt",30):"#7c3aed",("PoolAdapt",50):"#0891b2",("SAGE-SLO",20):"#059669",("SAGE-SLO",30):"#65a30d"}
    for m,b in keys:data[(m,b)]=[num(r,"reranking_latency_ms") for r in agg if r["method"]==m and r["operating_point"]==f"B{b}"]
    hi=max(max(v) for v in data.values()); edges=np.linspace(0,hi,31); counts={k:np.histogram(v,bins=edges)[0] for k,v in data.items()}; maxc=max(max(v) for v in counts.values())
    body=[]; L,T,R,B=90,70,950,555
    body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>'])
    for tick in range(0,maxc+1,max(1,math.ceil(maxc/4))):
        ty=B-(B-T)*tick/maxc;body.append(f'<line class="grid" x1="{L}" y1="{ty}" x2="{R}" y2="{ty}"/><text class="small" x="{L-8}" y="{ty+4}" text-anchor="end">{tick}</text>')
    body.extend([f'<text class="small" x="{L}" y="{B+18}" text-anchor="middle">0</text>',f'<text class="small" x="{R}" y="{B+18}" text-anchor="middle">{hi:.0f}</text>'])
    for li,(k,cs) in enumerate(counts.items()):
        pts=[]
        for bi,c in enumerate(cs):pts.append(f"{L+(R-L)*bi/30},{B-(B-T)*c/maxc}")
        legend_x=660+(li%2)*145;legend_y=48+(li//2)*18
        body.append(f'<polyline fill="none" stroke="{series_colors[k]}" stroke-width="2.5" points="{" ".join(pts)}"/><line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="{series_colors[k]}" stroke-width="3"/><text class="small" x="{legend_x+23}" y="{legend_y+4}">{k[0]} B{k[1]}</text>')
    render_svg(svg_base("Per-query reranking latency distributions","Latency bins (ms)","Query count per bin",''.join(body),1000,640),OUT/"figures/latency_distribution.png")
    # Cost consumption, K distribution; operating points are aggregate means and per-query box-like quantile bands.
    pts=[r for r in summaries if r["method"] in ("PoolAdapt","Fixed Prefix","SAGE-SLO","PACE-EF","Random")]
    body=[]; L,T,R,B=90,115,950,555; maxy=100; groups=sorted({r["operating_point"] for r in pts},key=lambda x:int(x[1:])); xgap=(R-L)/len(groups)
    for gi,op in enumerate(groups):
        rs=[r for r in pts if r["operating_point"]==op]
        for j,r in enumerate(rs):
            x=L+xgap*(gi+.5)+(j-(len(rs)-1)/2)*13; y=B-(B-T)*r["average_actual_pairs"]/maxy
            body.append(f'<rect x="{x-5}" y="{y}" width="10" height="{B-y}" fill="{colors[r["method"]]}"/><title>{r["method"]} {op}: {r["average_actual_pairs"]:.3f} pairs/query</title>')
        body.append(f'<text class="small" x="{L+xgap*(gi+.5)}" y="{B+20}" text-anchor="middle">{op}</text>')
    for ti,tick in enumerate((0,20,40,60,80,100)):
        ty=B-(B-T)*tick/maxy;body.append(f'<line class="grid" x1="{L}" y1="{ty}" x2="{R}" y2="{ty}"/><text class="small" x="{L-8}" y="{ty+4}" text-anchor="end">{tick}</text>')
    body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>',f'<text class="small" x="{L}" y="{T-12}">Full Rerank reference: 100 pairs/query; bars show actual pairs</text>'])
    for li,m in enumerate(METHODS[:-1]):
        lx=90+li*170;body.append(f'<rect x="{lx}" y="48" width="10" height="10" fill="{colors[m]}"/><text class="small" x="{lx+14}" y="57">{m}</text>')
    render_svg(svg_base("Reranking pair consumption","Operating point","Actual reranked pairs per query",''.join(body),1000,640),OUT/"figures/reranking_cost_comparison.png")
    # K distribution, query-level 10th/50th/90th percentile.
    body=[]; L,T,R,B=90,70,950,555; methods=METHODS[:-1]; ops=(10,20,30,50); maxy=max(110,math.ceil(max(num(r,"selected_k") for r in agg if r["method"]!="Full Rerank")/10)*10); step=(R-L)/(len(methods)*4)
    tickstep=max(10,math.ceil(maxy/5/10)*10)
    for tick in range(0,int(maxy)+1,tickstep):
        ty=B-(B-T)*tick/maxy;body.append(f'<line class="grid" x1="{L}" y1="{ty}" x2="{R}" y2="{ty}"/><text class="small" x="{L-8}" y="{ty+4}" text-anchor="end">{tick}</text>')
    for mi,m in enumerate(methods):
        for bi,b in enumerate(ops):
            vals=[num(r,"selected_k") for r in agg if r["method"]==m and r["operating_point"]==f"B{b}"]
            if not vals:continue
            x=L+(mi*4+bi+.5)*step; lo,md,hi=[pct(vals,q) for q in (10,50,90)]
            ylo=B-(B-T)*lo/maxy; ymd=B-(B-T)*md/maxy; yhi=B-(B-T)*hi/maxy
            body.append(f'<line x1="{x}" y1="{yhi}" x2="{x}" y2="{ylo}" stroke="{colors[m]}" stroke-width="3"/><line x1="{x-3}" y1="{yhi}" x2="{x+3}" y2="{yhi}" stroke="{colors[m]}"/><line x1="{x-3}" y1="{ylo}" x2="{x+3}" y2="{ylo}" stroke="{colors[m]}"/><circle cx="{x}" cy="{ymd}" r="3" fill="{colors[m]}"/><text class="small" x="{x}" y="{B+17}" text-anchor="middle">B{b}</text>')
        body.append(f'<text class="small" x="{L+(mi*4+2)*step}" y="{B+42}" text-anchor="middle">{m}</text>')
    body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>',f'<text class="small" x="{L}" y="{T-12}">Whiskers: P10–P90; point: median</text>'])
    render_svg(svg_base("Controller selected K distributions","Method grouped across B10, B20, B30, B50","Selected K (candidate count)",''.join(body),1000,640),OUT/"figures/k_distribution.png")
    # Failure examples plot ranks most extreme qualifying criteria; explanatory rows are authoritative.
    selected=[]
    for category in ("under-allocation candidate","over-allocation candidate"):
        group=[r for r in cases if r["failure_category"]==category]
        selected.extend(sorted(group,key=lambda r:(-abs(float(r["criterion_value"])),int(r["query_id"])))[:10])
    body=[]; L,T,R,B=90,70,950,555
    if selected:
        for i,r in enumerate(selected[:20]):
            y=T+i*22; catmax=max(abs(float(x["criterion_value"])) for x in selected if x["failure_category"]==r["failure_category"]) or 1; width=(R-L)*abs(float(r["criterion_value"])) / catmax
            width=(R-L-100)*abs(float(r["criterion_value"])) / catmax
            body.append(f'<rect x="{L}" y="{y}" width="{width}" height="14" fill="{"#dc2626" if "under" in r["failure_category"] else "#d97706"}"/><text class="small" x="{min(L+width+5,R-75)}" y="{y+12}">Q{r["query_id"]}</text>')
        body.extend(['<rect x="660" y="50" width="12" height="12" fill="#dc2626"/><text class="small" x="678" y="61">Under-allocation</text>','<rect x="810" y="50" width="12" height="12" fill="#d97706"/><text class="small" x="828" y="61">Over-allocation</text>'])
    else: body.append('<text class="label" x="500" y="300" text-anchor="middle">No cases met the predeclared quality/allocation thresholds.</text>')
    render_svg(svg_base("Validated per-query allocation failure candidates","Selected examples ordered by threshold magnitude","Within-category normalized criterion magnitude",''.join(body),1000,640),OUT/"figures/failure_case_examples.png")
    # Pareto scatters: annotation-free colors with clear efficient/dominated distinction.
    for obj,xfield,xlab,name in [("quality_vs_actual_pairs","average_actual_pairs","Average actual reranked pairs/query","quality_cost_pareto.png"),("quality_vs_reranking_latency","mean_reranking_latency_ms","Mean reranking latency (ms)","quality_latency_pareto.png")]:
        sub=[r for r in pareto_rows if r["pareto_objective"]==obj]; xmax=max(num(r,xfield) for r in sub)*1.1; xmin=0; ymin=0.0; ymax=1.0
        body=[]; L,T,R,B=90,105,950,555
        for li,m in enumerate(METHODS):
            lx=95+(li%3)*260;ly=55+(li//3)*18
            body.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{colors[m]}"/><text class="small" x="{lx+10}" y="{ly+4}">{m}</text>')
        body.append('<circle cx="800" cy="55" r="5" fill="#444"/><text class="small" x="810" y="59">Efficient: filled; dominated: open</text>')
        for ti in range(6):
            xval=xmax*ti/5; px=L+(R-L)*xval/xmax;body.append(f'<line class="grid" x1="{px}" y1="{T}" x2="{px}" y2="{B}"/><text class="small" x="{px}" y="{B+18}" text-anchor="middle">{xval:.0f}</text>')
            yval=ymin+(ymax-ymin)*ti/5; py=B-(B-T)*(yval-ymin)/(ymax-ymin);body.append(f'<line class="grid" x1="{L}" y1="{py}" x2="{R}" y2="{py}"/><text class="small" x="{L-8}" y="{py+4}" text-anchor="end">{yval:.2f}</text>')
        for r in sub:
            x=L+(R-L)*num(r,xfield)/xmax; y=B-(B-T)*(num(r,"ndcg_at_10")-ymin)/(ymax-ymin); c=colors[r["method"]]; fill=c if r["pareto_efficient"] else "white"
            body.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{fill}" stroke="{c}" stroke-width="2"><title>{r["method"]} {r["operating_point"]}; nDCG {r["ndcg_at_10"]:.3f}; efficient={r["pareto_efficient"]}</title></circle>')
        body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>'])
        render_svg(svg_base("Pareto frontier: quality and resource use",xlab,"nDCG@10",''.join(body),1000,640),OUT/"pareto"/name)
    # operating point comparison for all methods across quality and pair load.
    body=[]; L,T,R,B=90,125,950,555; xmax=max(r["average_actual_pairs"] for r in summaries)*1.1; ymin=0.0; ymax=1.0
    for ti in range(6):
        xv=xmax*ti/5;px=L+(R-L)*xv/xmax;yv=ymin+(ymax-ymin)*ti/5;py=B-(B-T)*(yv-ymin)/(ymax-ymin)
        body.append(f'<line class="grid" x1="{px}" y1="{T}" x2="{px}" y2="{B}"/><text class="small" x="{px}" y="{B+18}" text-anchor="middle">{xv:.0f}</text><line class="grid" x1="{L}" y1="{py}" x2="{R}" y2="{py}"/><text class="small" x="{L-8}" y="{py+4}" text-anchor="end">{yv:.2f}</text>')
    for li,m in enumerate(METHODS):
        lx=95+(li%3)*260;ly=55+(li//3)*18;body.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{colors[m]}"/><text class="small" x="{lx+10}" y="{ly+4}">{m}</text>')
    body.append('<text class="small" x="95" y="105">Markers: circle B10  square B20  triangle B30  diamond B50  plus B100</text>')
    for r in summaries:
        x=L+(R-L)*r["average_actual_pairs"]/xmax; y=B-(B-T)*(r["ndcg_at_10"]-ymin)/(ymax-ymin); c=colors[r["method"]]; op=r["operating_point"]
        if op=="B10": mark=f'<circle cx="{x}" cy="{y}" r="5" fill="{c}"/>'
        elif op=="B20": mark=f'<rect x="{x-4.5}" y="{y-4.5}" width="9" height="9" fill="{c}"/>'
        elif op=="B30": mark=f'<polygon points="{x},{y-6} {x-6},{y+5} {x+6},{y+5}" fill="{c}"/>'
        elif op=="B50": mark=f'<polygon points="{x},{y-6} {x-6},{y} {x},{y+6} {x+6},{y}" fill="{c}"/>'
        else: mark=f'<line x1="{x-6}" y1="{y}" x2="{x+6}" y2="{y}" stroke="{c}" stroke-width="3"/><line x1="{x}" y1="{y-6}" x2="{x}" y2="{y+6}" stroke="{c}" stroke-width="3"/>'
        body.append(mark)
    body.extend([f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/><line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>'])
    render_svg(svg_base("Operating-point overview","Average actual pairs/query","nDCG@10",''.join(body),1000,640),OUT/"pareto/operating_point_comparison.png")

def run()->None:
    for d in ["audit","tables","statistics","pareto","failure_analysis","latency_cost","figures","matched_budget"]:(OUT/d).mkdir(parents=True,exist_ok=True)
    raw=read_rows(INPUT); validation=validate(raw)
    manifest_paths=[INPUT,ROOT/"data/scifact/qrels/test.tsv",OUT/"audit/test_provenance_audit_report.md",OUT/"audit/artifact_consistency_audit.csv",OUT/"audit/data_handoff_specification.md",ROOT/"results/phase2/evaluation/final_gpu/phase2_summary.json"]
    manifest_files=[]
    for p in manifest_paths:
        count=len(raw) if p==INPUT else max(0,sum(1 for _ in p.open(encoding="utf-8-sig"))-1)
        manifest_files.append({"path":str(p.relative_to(ROOT)).replace("\\","/"),"sha256_current_bytes":sha(p),"rows":count,"hash_semantics":"current observed bytes; downstream hash, not a historical frozen-output hash"})
    manifest={"input_files":manifest_files,"analysis_configuration":{"random_seed":SEED,"bootstrap_replicates":BOOTSTRAPS,"permutation_replicates":PERMUTATIONS,"primary_matched_budgets":[10,20,50],"budget_tolerance_pct":MATCH_TOLERANCE_PCT,"random_seed_aggregation":"arithmetic mean of the five seed records within each query/method/operating point; n remains 300 queries","high_latency_rule":HIGH_LATENCY_RULE,"positive_qrels_validation_source":"three-column SciFact qrels; score > 0","latency_scope":"pair preparation + Cross-Encoder inference + sorting; excludes retrieval, selection, model load and warm-up"}}
    (OUT/"audit/downstream_input_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    report=["# Downstream handoff validation report","",f"Status: **{'PASS' if not validation['errors'] else 'FAIL'}**",f"Rows: {validation['rows']}; unique TEST queries: {validation['queries']}; unique composite keys: {validation['unique_composite_keys']}; strata: {validation['strata']}.",f"Methods: {', '.join(validation['methods'])}.",f"Operating points: {', '.join(validation['operating_points'])}.",f"Random seeds: {', '.join(validation['random_seeds'])}; cross-query aggregation averages seeds within query.",f"Positive qrels: {validation['positive_qrels_judgments']} judgments across {validation['positive_qrels_queries']} query IDs; TEST IDs missing positive judgments: {len(validation['positive_qrels_coverage_missing'])}.",f"Execution-manifest linkage values: {len(validation['manifest_hashes'])} unique hash(es).",f"Missing-value counts: `{json.dumps(validation['missing_by_column'],sort_keys=True)}`.",f"Metric ranges: `{json.dumps(validation['metric_ranges'],sort_keys=True)}`.","", "Critical checks: full query coverage in each stratum; unique composite keys; selected_k equals actual pairs; finite nonnegative latency; metric range [0,1]; positive qrels coverage; execution manifest hash populated.","", "The handoff's blank controller features, query groups, and execution IDs are unavailable values, not zeros. This downstream hash records the current observed input bytes; it does not claim a historical frozen-output hash.",""]
    if validation["errors"]: report += ["## Validation errors", *[f"- {e}" for e in validation["errors"]]]
    else: report += ["All critical structural, range, coverage, pair-count, and linkage checks passed."]
    (OUT/"audit/downstream_validation_report.md").write_text("\n".join(report),encoding="utf-8")
    if validation["errors"]: raise SystemExit("Critical Phase 0 validation failed; downstream analysis stopped. See audit/downstream_validation_report.md")
    agg=aggregate(raw); summaries=summarize(agg); seed_sensitivity=random_seed_sensitivity(raw); source_check=compare_source_aggregates(summaries,raw)
    if source_check["mismatches"]: raise SystemExit(f"Source aggregate reproduction failed: {source_check['mismatches'][:5]}")
    with (OUT/"audit/downstream_validation_report.md").open("a",encoding="utf-8") as f:f.write(f"\n\nSource aggregate comparison: PASS for the quality, pair-count, mean-latency, and non-Random percentile fields. {source_check['not_compared_values']} historical Random P50/P95 fields use a different aggregation and are not used as primary validation targets. {source_check['not_compared_note']}\n")
    # Preserve source input; query-level output is represented by validated seed-aggregated derived data.
    write_csv(OUT/"matched_budget/query_level_analysis.csv",agg)
    fields=list(summaries[0]);write_csv(OUT/"tables/main_results_summary.csv",summaries,fields)
    write_csv(OUT/"statistics/random_seed_sensitivity.csv",seed_sensitivity)
    matched=[]
    for b in (10,20,30,50):
        pa=next(x for x in summaries if x["method"]=="PoolAdapt" and x["operating_point"]==f"B{b}"); fx=next(x for x in summaries if x["method"]=="Fixed Prefix" and x["operating_point"]==f"B{b}")
        dd=100*(pa["average_actual_pairs"]-fx["average_actual_pairs"])/fx["average_actual_pairs"]
        for x in (pa,fx): matched.append({**x,"pair_budget_difference_pct":dd,"within_match_tolerance":abs(dd)<=MATCH_TOLERANCE_PCT,"primary_comparison":b in (10,20,50),"comparison_status":"primary matched" if b in (10,20,50) else "supplementary; fails ±5% matching criterion"})
    write_csv(OUT/"matched_budget/matched_budget_results.csv",matched)
    write_csv(OUT/"tables/matched_budget_summary.csv",matched)
    statrows,stat_summary=paired_stats(agg);write_csv(OUT/"statistics/paired_statistical_analysis.csv",statrows);write_csv(OUT/"tables/paired_statistical_summary.csv",stat_summary)
    pfront=pareto(summaries);write_csv(OUT/"pareto/pareto_analysis.csv",pfront)
    latrows=latency(agg);write_csv(OUT/"latency_cost/latency_cost_summary.csv",latrows)
    cases=failures(agg);write_csv(OUT/"failure_analysis/failure_case_analysis.csv",cases)
    figures(summaries,agg,latrows,pfront,cases)
    # Query-group Analysis is supplementary and lives under query_group_analysis;
    # it does not populate the frozen TEST controller fields.
    (OUT/"statistics/statistical_analysis_report.md").write_text(stat_report(statrows),encoding="utf-8")
    (OUT/"failure_analysis/failure_analysis_report.md").write_text(failure_report(cases),encoding="utf-8")
    (OUT/"audit/downstream_consistency_audit.md").write_text(consistency_report(raw,agg,summaries,matched,statrows,pfront,cases,validation,source_check),encoding="utf-8")
    print(f"PASS: {len(raw)} raw rows, {validation['queries']} queries, {len(agg)} query-aggregated rows, {len(statrows)} paired tests, {len(pfront)} Pareto rows, {len(cases)} failure candidates")

def stat_report(rows:list[dict[str,Any]])->str:
    lines=["# Paired statistical analysis","","Paired comparisons use the same 300 frozen TEST query IDs. PoolAdapt and Fixed Prefix are aligned by query before all tests. Random executions are averaged within each query before any cross-query analysis; Random therefore contributes 300 query-level observations, not 1,500 independent queries. The primary budget set is B10, B20, B50; B30 is excluded because actual pair means differ by more than ±5%.","","Bootstrap intervals use 10,000 query-level paired bootstrap resamples (seed 20261112). The two-sided paired permutation test independently sign-flips query differences using 100,000 Monte Carlo replicates and a plus-one p-value. Holm correction is applied to three operating points separately for each metric. Wilcoxon signed-rank is supplementary with exact zero differences excluded. Cohen's dz is mean paired difference divided by sample SD; it is undefined if SD=0. A non-significant result is not evidence of equivalence.","","| Budget | Metric | Mean difference | 95% bootstrap CI | Raw p | Holm p | Cohen dz |","|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['operating_point']} | {r['metric']} | {r['mean_paired_difference']:.5f} | [{r['bootstrap_95_ci_low']:.5f}, {r['bootstrap_95_ci_high']:.5f}] | {r['permutation_p_raw']:.5g} | {r['permutation_p_holm_adjusted']:.5g} | {fmt(r['cohen_dz'],4)} |")
    lines += ["","Latency and pair-count associations are descriptive Spearman correlations and do not establish causality. See `latency_cost/latency_cost_summary.csv`. A later supplementary Query-group Analysis is documented under `query_group_analysis/`; it uses reconstructed descriptive groups and does not populate the original controller fields or validate an adaptive policy.",""]
    return "\n".join(lines)

def failure_report(cases:list[dict[str,Any]])->str:
    n=defaultdict(int)
    for x in cases:n[x["failure_category"]]+=1
    return "\n".join(["# Failure analysis","","Thresholds were defined before selecting cases. Under-allocation candidates require fewer actual pairs than same-budget Fixed Prefix and an nDCG@10 decrease of at least 0.20. Over-allocation candidates require at least 10% and at least two additional actual pairs, with no nDCG@10 gain. Allocation categories are assessed only at primary matched budgets B10/B20/B50. All qualifying records are retained.","",f"Validated under-allocation candidates: {n['under-allocation candidate']}.",f"Validated over-allocation candidates: {n['over-allocation candidate']}.",f"Representative high-latency cases included: {n['high-latency case']} (top three Tukey outliers per PoolAdapt B20, B30, and B50).","","No TEST controller features or sparse/dense pool signals are available, so signal-related mechanisms cannot be tested. These are descriptive observations, not demonstrated causal failures. Full Rerank metrics are included per query for context. The CSV contains query text, paired metrics, counts, criteria, and evidence limitations.",""])

def consistency_report(raw,agg,summaries,matched,statrows,pfront,cases,val,source_check)->str:
    grouped=defaultdict(set)
    for row in agg: grouped[(row["method"],row["operating_point"])].add(row["query_id"])
    primary_ok=all(len(grouped[(m,f"B{b}")])==300 for m in ("PoolAdapt","Fixed Prefix") for b in (10,20,50))
    return "\n".join(["# Downstream consistency audit","", "## Checks",f"- Phase 0 validation: {'PASS' if not val['errors'] else 'FAIL'} ({val['rows']} execution rows; {val['queries']} unique queries).",f"- Source aggregate comparison: {'PASS' if not source_check['mismatches'] else 'FAIL'} ({source_check['matched_rows']} rows; {source_check['compared_values']} core values at 1e-10; {source_check['not_compared_values']} Random P50/P95 fields explicitly excluded).",f"- Primary matched-budget comparisons use 300 query-level observations: {'PASS' if primary_ok else 'FAIL'}.","- Random seed aggregation is within query; no 1,500-query inference.","- B30 is supplementary and marked unmatched where its pair-count difference exceeds ±5%.","- Compression ratio and pair reduction use actual reranked-pair means relative to Full Rerank; labels avoid monetary/GPU costs.","- Latency fields are reranking-stage latency, not end-to-end latency.","- Paired tests align by query, report bootstrap CIs, permutation p-values and per-metric Holm adjustment.","- Pareto objective uses quality versus pairs and quality versus latency separately.",f"- Failure cases: {len(cases)} threshold-qualified descriptive candidates; feature mechanisms remain unavailable.","- A later supplementary Query-group Analysis uses reconstructed descriptive groups; frozen TEST controller feature vectors remain unavailable and no adaptive policy is validated.","- Provenance limitations from upstream audit remain: two FAIL and three NOT VERIFIED; current input hashes do not establish historical output hashes.","- Raw query-level input preserved; no Phase 2 predictions, qrels, checkpoints, configurations, manifests, or other frozen source outputs were modified.","","## Generated population",f"- Raw executions: {len(raw)}",f"- Query-level analysis rows after within-query Random seed averaging: {len(agg)}",f"- Paired tests: {len(statrows)}",f"- Pareto rows: {len(pfront)}",f"- Failure candidates: {len(cases)}","","See the generated CSVs and `audit/downstream_input_manifest.json` for exact values and hashes.",""])

if __name__=="__main__":run()
