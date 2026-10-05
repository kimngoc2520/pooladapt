"""Create the separate candidate relevance-label artifact from qrels."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from src.data import load_beir_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset-dir',type=Path,default=Path('data/scifact')); p.add_argument('--split',choices=('train','validation','test'),required=True); p.add_argument('--candidates',type=Path,default=Path('results/phase2/01_candidate_logging/candidate_features.csv')); p.add_argument('--output',type=Path,default=Path('results/phase2/02_candidate_labels/candidate_labels.csv')); a=p.parse_args()
    _,_,qrels=load_beir_dataset(a.dataset_dir,a.split); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.candidates.open(encoding='utf-8-sig',newline='') as f: candidates=list(csv.DictReader(f))
    by_query={}
    for c in candidates: by_query[c['query_id']]=by_query.get(c['query_id'],0)+1
    if set(by_query)!=set(qrels): raise ValueError(f'candidate query IDs do not match qrels; missing={len(set(qrels)-set(by_query))}, extra={len(set(by_query)-set(qrels))}')
    wrong={qid:n for qid,n in by_query.items() if n!=100}
    if wrong: raise ValueError(f'expected exactly 100 candidates per query; examples={list(wrong.items())[:5]}')
    seen=set(); rows=[]
    for c in candidates:
        key=(c['query_id'],c['doc_id'])
        if key in seen: raise ValueError(f'duplicate candidate pair: {key}')
        seen.add(key); rows.append({'query_id':key[0],'doc_id':key[1],'is_relevant':int(qrels.get(key[0],{}).get(key[1],0)>0),'label_source':'qrels'})
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['query_id','doc_id','is_relevant','label_source']); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} labels to {a.output}')
if __name__=='__main__': main()
