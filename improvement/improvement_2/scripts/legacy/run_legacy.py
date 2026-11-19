"""Development-only CT2.0--CT2.2 execution compliant with Protocol v8.2.

Consumes persisted fixed-prefix results; it never loads TEST or runs a reranker.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import scipy, sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, mrr_at_k
B=(10,20,30,50,100); COST=(11,12,14,17,20); SEED=20261117; DRAWS=2000; PERMS=10000
Q=("query_length","num_numeric_tokens","avg_idf","max_idf")
P=("top1_top2_rrf_margin","top20_overlap","rank_correlation_union","score_entropy","score_gini","redundancy","concentration","sparse_dense_disagreement")
CONFIG={"full":Q+P,"query_only":Q,"pool_only":P}

def rows(p:Path, sep=","): return list(csv.DictReader(p.open(encoding="utf-8-sig",newline=""),delimiter=sep))
def write(p:Path, xs:list[dict]):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(xs[0]) if xs else ["status"]); w.writeheader(); w.writerows(xs)
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def boot(x, seed=SEED):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); m=rng.choice(x,(DRAWS,len(x)),replace=True).mean(1)
    return float(x.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))
def signflip(x,seed):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); obs=abs(x.mean()); vals=(rng.choice((-1,1),(PERMS,len(x)))*x).mean(1)
    return float((1+(np.abs(vals)>=obs).sum())/(PERMS+1))
def holm(ps):
    out=[0.]*len(ps); prev=0.
    for r,i in enumerate(np.argsort(ps)):
        prev=max(prev,min(1.,(len(ps)-r)*ps[i])); out[i]=prev
    return out
def model(c,seed): return Pipeline((("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("lr",LogisticRegression(C=c,max_iter=1000,class_weight="balanced",random_state=seed))))
def align(m,x):
    raw=m.predict_proba(x); out=np.zeros((len(x),5)); avail=np.zeros(5,bool)
    for j,k in enumerate(m.named_steps["lr"].classes_): out[:,B.index(int(k))]=raw[:,j]; avail[B.index(int(k))]=True
    return out,avail
def hard(p): return 1-p[:,0]
def strata(y): return np.array([0 if z==10 else 1 if z in(20,30) else 2 for z in y])
def select_c(x,y,seed):
    inner=StratifiedKFold(3,shuffle=True,random_state=seed); st=strata(y); allout=[]; best=None
    for c in (.01,.1,1.,10.):
        pr=np.zeros((len(y),5)); mask=np.zeros(len(y),bool)
        for tr,va in inner.split(x,st):
            m=model(c,seed);m.fit(x[tr],y[tr]); pr[va],_=align(m,x[va]);mask[va]=True
        ap=average_precision_score(y>10,hard(pr)); ll=log_loss(y,pr,labels=B)
        allout.append((c,float(ap),float(ll)))
    mx=max(t[1] for t in allout); chosen=min(t[0] for t in allout if np.isclose(t[1],mx))
    return chosen,allout
def metric_table(pred,y):
    htrue=y>10; hp=hard(pred); k=np.array([B[i] for i in pred.argmax(1)])
    out={"hard_pr_auc":float(average_precision_score(htrue,hp)),"hard_prevalence":float(htrue.mean()),"hard_precision":float(precision_recall_fscore_support(htrue,k>10,zero_division=0)[0][1]),"hard_recall":float(precision_recall_fscore_support(htrue,k>10,zero_division=0)[1][1]),"three_group_macro_f1":float(f1_score(np.select([y==10,np.isin(y,(20,30))],[0,1],default=2),np.select([k==10,np.isin(k,(20,30))],[0,1],default=2),average="macro",zero_division=0)),"five_class_macro_f1":float(f1_score(y,k,labels=B,average="macro",zero_division=0))}
    return out,k
def bisect_policy(prob,available,target):
    logs=np.log(np.maximum(prob,1e-12)); logs[~available]=-np.inf; g=-(np.array(B)-np.mean(B))/np.std(B)
    lo,hi=-100.,100.; seen=[]
    for _ in range(80):
        z=(lo+hi)/2; k=np.array(B)[np.argmax(logs+z*g,1)]; mean=k.mean();seen.append((z,k,mean))
        if mean>target: lo=z
        else: hi=z
    # Discrete ties: closest cost, then lower realized K.
    z,k,m=min(seen,key=lambda t:(abs(t[2]-target),t[2],abs(t[0])))
    return k,float(z),float(m),abs(m-target)<=.1
def null_diff(k, vals, rng):
    pi=np.array([(k==b).mean() for b in B]); actual=np.array([vals[i,B.index(z)] for i,z in enumerate(k)])
    null=(vals*pi).sum(1); return actual-null
def frontier(prob,av,vals):
    ans=[]; records=[]
    for c in COST:
        k,lam,mean,ok=bisect_policy(prob,av,c)
        d=null_diff(k,vals,None); mu,lo,hi=boot(d,SEED+c); records.append({"target_average_K":c,"lambda":lam,"average_K":mean,"feasible":ok,"gain":mu,"ci95_low":lo,"ci95_high":hi,"signflip_p":signflip(d,SEED+c)})
        ans.append((k,d,ok))
    return ans,records
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-root",type=Path,default=ROOT/"improvement/improvement_2");ap.add_argument("--protocol",type=Path,required=True);a=ap.parse_args()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); out=a.output_root/"results"/f"ct2_v82_development_{stamp}"; out.mkdir(parents=True)
    ids=[x.strip() for p in (ROOT/"data/scifact/splits/train_query_ids.txt",ROOT/"data/scifact/splits/validation_query_ids.txt") for x in p.read_text().splitlines() if x.strip()]
    qrels={}
    for p in (ROOT/"data/scifact/qrels/train.tsv",ROOT/"data/scifact/qrels/validation.tsv"):
        for r in rows(p,"\t"):qrels.setdefault(r["query-id"],{})[r["corpus-id"]]=int(r["score"])
    lab={r["query_id"]:int(r["M_star"]) for r in rows(ROOT/"results/phase2/02_oracle/mstar_labels_tau099.csv")}; y=np.array([lab[i] for i in ids])
    pf={r["query_id"]:r for r in rows(ROOT/"results/phase2/01_candidate_logging/pool_features.csv")}; qf={r["query_id"]:r for r in rows(ROOT/"results/phase2/01_candidate_logging/query_features.csv")}
    vals=np.zeros((len(ids),5,3)); latency=np.full((len(ids),5),np.nan)
    for source in (ROOT/"results/phase2/development_reranking/train/reranking_results.json",ROOT/"results/phase2/development_reranking/validation/reranking_results.json"):
        methods=json.loads(source.read_text())["methods"]
        for j,b in enumerate(B):
            for r in methods[f"fixed_prefix_{b}"]["predictions"]:
                i=ids.index(str(r["query_id"])); ranked=[str(x) for x in r["document_ids"]]; rel=qrels[ids[i]]; vals[i,j]=[ndcg_at_k(ranked,rel),recall_at_k(ranked,rel),mrr_at_k(ranked,rel)]; latency[i,j]=float(r.get("latency_rerank_seconds","nan"))
    # CT2.0
    c0=out/"ct2_0_oracle_headroom"; rec=[]
    ix=np.array([B.index(z) for z in y]); d=vals[np.arange(len(ids)),ix,0]-vals[:,0,0]; mu,lo,hi=boot(d)
    rec.append({"analysis":"oracle_minus_fixed10","n":len(ids),"estimate":mu,"ci95_low":lo,"ci95_high":hi,"delta_A":.005,"gate_A":"above" if lo>.005 else "below" if hi<.005 else "inconclusive"});write(c0/"gate_a.csv",rec)
    env=[]
    for j,b in enumerate(B):
        gap=vals[:,j,0]-vals[:,-1,0];env.append({"K":b,"prefix_exceeds_full_queries":int((gap>0).sum()),"mean_signed_gap":float(gap.mean()),"max_prefix_ndcg_mean":float(vals[:,j,0].mean())})
    write(c0/"full_prefix_envelope.csv",env)
    p=(np.mean(y)-10)/10; mix=(1-p)*vals[:,0]+p*vals[:,1];write(c0/"fixed10_20_mixture.csv",[{"expected_average_K":10+10*p,"mixture_probability_fixed20":p,**{f"mean_{n}":float(mix[:,j].mean()) for j,n in enumerate(("nDCG10","recall10","mrr10"))}}])
    # CV
    folds=[]; aggregate={}; innerrows=[]; foldrows=[]
    for name,fs in CONFIG.items():
        x=np.array([[float(({**pf[i],**qf[i]})[z]) if ({**pf[i],**qf[i]})[z] else np.nan for z in fs] for i in ids]); pr=defaultdict(list); masks=defaultdict(list)
        for rep in range(10):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=rep).split(x,strata(y))):
                c,diag=select_c(x[tr],y[tr],rep*100+fold); m=model(c,rep*100+fold);m.fit(x[tr],y[tr]); pp,aa=align(m,x[te])
                for cc,score,ll in diag: innerrows.append({"configuration":name,"repeat":rep,"fold":fold,"C":cc,"pooled_inner_hard_pr_auc":score,"multiclass_log_loss":ll,"selected":cc==c})
                for z,pos in enumerate(te): pr[pos].append(pp[z]);masks[pos].append(aa);foldrows.append({"configuration":name,"repeat":rep,"fold":fold,"query_id":ids[pos],"selected_C":c,"class_available":"|".join(map(str,aa.astype(int)))})
        pp=np.array([np.mean(pr[i],0) for i in range(len(ids))]); aa=np.array([np.all(masks[i],0) for i in range(len(ids))]); aggregate[name]=(pp,aa)
    write(out/"ct2_1_feature_ablation"/"inner_selection.csv",innerrows);write(out/"ct2_1_feature_ablation"/"folds.csv",foldrows)
    met=[]; predrows=[]; fronts={}
    for name,(pp,aa) in aggregate.items():
        mt,k=metric_table(pp,y);met += [{"configuration":name,"measure":z,"value":v} for z,v in mt.items()]
        fr,frrows=frontier(pp,aa,vals[:,:,0]);fronts[name]=fr;write(out/"ct2_1_feature_ablation"/f"frontier_{name}.csv",frrows)
        for i in range(len(ids)):predrows.append({"configuration":name,"query_id":ids[i],"M_star":int(y[i]),"predicted_K":int(k[i]),"class_available":"|".join(map(str,aa[i].astype(int))),**{f"p_{b}":float(pp[i,j]) for j,b in enumerate(B)}})
    write(out/"ct2_1_feature_ablation"/"prediction_metrics.csv",met);write(out/"ct2_1_feature_ablation"/"oof_predictions.csv",predrows)
    # CT2.2 from compliant Full OOF
    pp,aa=aggregate["full"]; _,k=metric_table(pp,y); c2=out/"ct2_2_controller_diagnosis"; rs=[]
    for i,q in enumerate(ids):
        j=B.index(int(k[i]));o=B.index(int(y[i]));rs.append({"query_id":q,"M_star":int(y[i]),"predicted_K":int(k[i]),"hard_true":int(y[i]>10),"hard_score":float(hard(pp[i:i+1])[0]),"hard_predicted":int(k[i]>10),"signed_K_error":int(k[i]-y[i]),"absolute_K_error":int(abs(k[i]-y[i])),"ordinal_index_error":j-o,"underallocated":int(k[i]<y[i]),"overallocated":int(k[i]>y[i]),"nDCG10":float(vals[i,j,0]),"Recall10":float(vals[i,j,1]),"MRR10":float(vals[i,j,2]),"nDCG_loss_vs_full":float(vals[i,-1,0]-vals[i,j,0]),"nDCG_loss_vs_oracle":float(vals[i,o,0]-vals[i,j,0]),"additional_pairs_vs_oracle":int(k[i]-y[i])})
    write(c2/"allocation_errors.csv",rs); cm=confusion_matrix(y,k,labels=B);write(c2/"five_class_confusion.csv",[{"actual":int(B[i]),**{f"pred_{b}":int(cm[i,j]) for j,b in enumerate(B)} } for i in range(5)])
    m=metric_table(pp,y)[0];write(c2/"hard_detection.csv",[{"measure":z,"value":v} for z,v in m.items()])
    manifest={"run_id":out.name,"protocol_version":"v8.2","protocol_path":str(a.protocol),"protocol_sha256":sha(a.protocol),"population":"SciFact TRAIN+VALIDATION","queries":len(ids),"test_used":False,"features":{x:list(z) for x,z in CONFIG.items()},"inner_objective":"pooled inner-OOF Hard PR-AUC","inner_secondary":"multiclass log-loss","outer_cv":"RepeatedStratifiedKFold 5x10 seeds 0..9","bootstrap":{"draws":DRAWS,"seed":SEED},"permutations":{"draws":PERMS,"seed":SEED},"cost_grid":COST,"code_sha256":sha(Path(__file__)),"environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"platform":platform.platform()}}
    (out/"reproducibility_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"run_id":out.name,"output":str(out),"status":"completed"}))
if __name__=="__main__":main()
