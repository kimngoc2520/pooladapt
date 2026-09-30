"""Extract inference-time SciFact query, candidate, and pool features."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data import load_beir_inputs
from src.retrieval import BM25Retriever, DenseRetriever, DEFAULT_MODEL, fuse_ranked_lists
from src.retrieval.bm25 import tokenize

N = 100

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def gini(values):
    x=np.sort(np.asarray(values, dtype=float)); total=x.sum()
    return float((2*np.dot(np.arange(1,len(x)+1),x)/(len(x)*total))-(len(x)+1)/len(x)) if total else 0.0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,default=Path('data/scifact'))
    p.add_argument('--model',default=DEFAULT_MODEL); p.add_argument('--queries',type=int,default=0)
    p.add_argument('--query-ids',type=Path,default=Path('results/phase1/per_query_analysis.csv'),help='Phase 1 query_id list; avoids reading qrels in feature extraction.')
    p.add_argument('--output-dir',type=Path,default=Path('results/phase2/01_candidate_logging'))
    args=p.parse_args()
    corpus, queries = load_beir_inputs(args.dataset_dir)
    bm25, dense = BM25Retriever(corpus), DenseRetriever(corpus, model_name=args.model)
    query_rows=[]; candidate_rows=[]; pool_rows=[]
    if args.query_ids.exists():
        with args.query_ids.open(encoding='utf-8-sig',newline='') as f: ids=[r['query_id'] for r in csv.DictReader(f)]
        missing=set(ids)-set(queries)
        if missing: raise ValueError(f'query IDs absent from queries file: {sorted(missing)[:5]}')
        query_items=[(qid,queries[qid]) for qid in ids]
    else: query_items=list(queries.items())
    query_items=query_items[:args.queries or None]
    for qid, query in query_items:
        b=bm25.retrieve(query, N); d=dense.retrieve(query,N)
        pool=fuse_ranked_lists({'bm25':b,'dense':d},top_n=N)
        if len(pool)!=N: raise ValueError(f'{qid}: expected {N} candidates, got {len(pool)}')
        if len({x['id'] for x in pool})!=N: raise ValueError(f'{qid}: duplicate candidate ids')
        bm={x['id']:x for x in b}; de={x['id']:x for x in d}
        for c in pool:
            c['bm25_rank']=bm.get(c['id'],{}).get('rank'); c['bm25_score']=bm.get(c['id'],{}).get('score')
            c['dense_rank']=de.get(c['id'],{}).get('rank'); c['dense_score']=de.get(c['id'],{}).get('score')
        for i,c in enumerate(pool):
            br,dr=c['bm25_rank'],c['dense_rank']
            candidate_rows.append({'query_id':qid,'doc_id':c['id'],'bm25_rank':br,'bm25_score':c['bm25_score'],'dense_rank':dr,'dense_score':c['dense_score'],'rrf_rank':c['rank'],'rrf_score':c['rrf_score'],'local_rrf_margin':c['rrf_score']-pool[i+1]['rrf_score'] if i+1<len(pool) else None,'rank_displacement':abs(br-dr) if br is not None and dr is not None else None})
        # Features with unavailable inputs are explicitly left blank.
        scores=np.asarray([c['rrf_score'] for c in pool],dtype=float); probs=scores/scores.sum()
        topb={x['id'] for x in b[:20]}; topd={x['id'] for x in d[:20]}
        union=set(bm)|set(de); paired=[(bm[x]['rank'],de[x]['rank']) for x in union if x in bm and x in de]
        rho=float(np.corrcoef(np.asarray(paired).T)[0,1]) if len(paired)>1 and np.std(np.asarray(paired),axis=0).min()>0 else None
        disps=[abs(c['bm25_rank']-c['dense_rank']) for c in pool if c['bm25_rank'] is not None and c['dense_rank'] is not None]
        norm=float(np.mean(disps)/99) if disps else None
        tokenized=[set(tokenize(corpus[c['id']].get('title','')+' '+corpus[c['id']].get('text',''))) for c in pool[:20]]
        jac=[]
        for i in range(len(tokenized)):
            for j in range(i):
                u=tokenized[i]|tokenized[j]
                if u: jac.append(len(tokenized[i]&tokenized[j])/len(u))
        pool_rows.append({'query_id':qid,'top1_top2_rrf_margin':scores[0]-scores[1],'top20_overlap':len(topb&topd)/20,'rank_correlation_union':rho,'score_entropy':float(-(probs*np.log(probs)).sum()),'score_gini':gini(scores),'redundancy':float(np.mean(jac)) if jac else None,'concentration':float(scores[:10].sum()/scores.sum()) if scores.sum() else None,'sparse_dense_agreement':1-norm if norm is not None else None,'sparse_dense_disagreement':norm})
        toks=tokenize(query); idfs=[bm25._index.idf.get(t) for t in toks if bm25._index.idf.get(t) is not None]
        query_rows.append({'query_id':qid,'query_length':len(toks),'num_numeric_tokens':sum(t.isnumeric() for t in toks),'entity_count':'','avg_idf':float(np.mean(idfs)) if idfs else None,'max_idf':max(idfs) if idfs else None})
    out=args.output_dir
    write_csv(out/'candidate_features.csv',candidate_rows,['query_id','doc_id','bm25_rank','bm25_score','dense_rank','dense_score','rrf_rank','rrf_score','local_rrf_margin','rank_displacement'])
    write_csv(out/'pool_features.csv',pool_rows,['query_id','top1_top2_rrf_margin','top20_overlap','rank_correlation_union','score_entropy','score_gini','redundancy','concentration','sparse_dense_agreement','sparse_dense_disagreement'])
    write_csv(out/'query_features.csv',query_rows,['query_id','query_length','num_numeric_tokens','entity_count','avg_idf','max_idf'])
    hashes={name:hashlib.sha256((args.dataset_dir/name).read_bytes()).hexdigest() for name in ('corpus.jsonl','queries.jsonl')}
    (out/'metadata.json').write_text(json.dumps({'dataset_dir':str(args.dataset_dir),'query_id_source':str(args.query_ids) if args.query_ids.exists() else 'all queries.jsonl rows','input_sha256':hashes,'query_count':len(query_rows),'candidate_rows':len(candidate_rows),'candidate_pool_size':N,'retrievers':['src.retrieval.BM25Retriever','src.retrieval.DenseRetriever'],'fusion':'src.retrieval.fuse_ranked_lists','rrf_k':60,'dense_model':args.model,'qrels_read_by_feature_extractor':False,'entity_count':'unavailable; blank','redundancy':'mean pairwise token-set Jaccard over RRF top 20','rank_correlation_union':'Pearson correlation of source ranks for candidates present in both Top-100 lists; null if undefined'},indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(candidate_rows)} candidates for {len(query_rows)} queries to {out}')
if __name__=='__main__': main()
