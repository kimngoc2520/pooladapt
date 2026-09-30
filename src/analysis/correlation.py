"""Query-level Spearman associations with supplied M* labels (descriptive only)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

TAUS=("095","098","099")

def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving original order."""
    p=np.asarray(p_values,dtype=float); n=len(p); order=np.argsort(p); adjusted=np.empty(n); running=0.0
    for rank,index in enumerate(order):
        running=max(running,(n-rank)*p[index]); adjusted[index]=min(1.0,running)
    return adjusted.tolist()

def analyze(features: dict[str,dict[str,list[dict]]], labels: list[dict], tau: str):
    label_column=next((x for x in (f'm_star_tau{tau}',f'mstar_tau{tau}',f'tau{tau}', 'm_star' if tau=='095' else '') if x and labels and x in labels[0]),None)
    if not label_column: raise ValueError(f'M* labels need a column for tau={tau}, e.g. m_star_tau{tau}')
    y={r['query_id']:float(r[label_column]) for r in labels if r.get(label_column,'')!=''}
    results=[]
    for level, by_query in features.items():
        names=sorted({name for row in by_query.values() for name,val in row.items() if name!='query_id' and val is not None})
        for name in names:
            pairs=[(float(row[name]),y[qid]) for qid,row in by_query.items() if qid in y and row.get(name) is not None and np.isfinite(float(row[name])) and np.isfinite(y[qid])]
            if len(pairs)<3: rho,p=0.0,1.0
            else:
                rho,p=spearmanr([a for a,_ in pairs],[b for _,b in pairs],nan_policy='omit')
                if not np.isfinite(rho) or not np.isfinite(p): rho,p=0.0,1.0
            results.append({'feature_name':name,'feature_level':level,'tau':f'0.{tau}','spearman_rho':float(rho),'p_value':float(p),'abs_rho':abs(float(rho)),'_n':len(pairs)})
    adjusted=holm_adjust([r['p_value'] for r in results])
    for r,p in zip(results,adjusted): r['p_holm']=p; r['selected']=r['abs_rho']>0.2 and p<0.05
    return results

def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def numeric_rows(rows, exclude=()):
    by={}
    for r in rows:
        row={'query_id':str(r['query_id'])}
        for k,v in r.items():
            if k in ('query_id',*exclude) or v in ('',None): continue
            try: row[k]=float(v)
            except (TypeError,ValueError): continue
        by[row['query_id']]=row
    return by

def load_feature_sets(candidate_path,pool_path,query_path):
    candidates=read_rows(candidate_path); grouped={}
    excluded={'doc_id'}
    for r in candidates:
        q=str(r['query_id']); grouped.setdefault(q,[])
        grouped[q].append({k:float(v) for k,v in r.items() if k not in ('query_id','doc_id') and v not in ('',None)})
    candidate_summary={}
    for q,rows in grouped.items():
        keys=set.intersection(*(set(r) for r in rows)) if rows else set(); candidate_summary[q]={'query_id':q}
        for k in keys: candidate_summary[q][f'{k}_mean']=float(np.mean([r[k] for r in rows]))
    return {'candidate':candidate_summary,'pool':numeric_rows(read_rows(pool_path)),'query':numeric_rows(read_rows(query_path),exclude=('entity_count',))}

def multicollinearity(features, threshold=.8):
    rows=[]
    for level, data in features.items():
        names=sorted({k for r in data.values() for k in r if k!='query_id'})
        for i,a in enumerate(names):
            for b in names[i+1:]:
                pairs=[(r[a],r[b]) for r in data.values() if a in r and b in r]
                if len(pairs)>2:
                    corr=np.corrcoef(np.asarray(pairs).T)[0,1]
                    if np.isfinite(corr) and abs(corr)>=threshold: rows.append({'feature_level':level,'feature_a':a,'feature_b':b,'pearson_r':float(corr)})
    return rows
