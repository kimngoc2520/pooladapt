"""Run descriptive feature--M* association analysis with Holm correction."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from src.analysis.correlation import analyze,load_feature_sets,multicollinearity,read_rows

FIELDS=['feature_name','feature_level','tau','spearman_rho','p_value','p_holm','abs_rho','selected']
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--features-dir',type=Path,default=Path('results/phase2/01_candidate_logging')); p.add_argument('--mstar-labels',type=Path,required=True,help='Query-level CSV with query_id and m_star_tau095/098/099 columns.'); p.add_argument('--output-dir',type=Path,default=Path('results/phase2/03_analysis')); a=p.parse_args()
    features=load_feature_sets(a.features_dir/'candidate_features.csv',a.features_dir/'pool_features.csv',a.features_dir/'query_features.csv'); labels=read_rows(a.mstar_labels); a.output_dir.mkdir(parents=True,exist_ok=True)
    for tau in ('095','098','099'):
        rows=analyze(features,labels,tau); path=a.output_dir/f'feature_mstar_analysis_tau{tau}.csv'
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows({k:r[k] for k in FIELDS} for r in rows)
        selected=[r for r in rows if r['selected']]
        print(f'tau=0.{tau}: G2 {"PASS" if selected else "FAIL"} ({len(selected)} selected features)')
    highcorr=multicollinearity(features); path=a.output_dir/'multicollinearity.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['feature_level','feature_a','feature_b','pearson_r']); w.writeheader(); w.writerows(highcorr)
    print(f'Flagged {len(highcorr)} feature pairs with |Pearson r| >= 0.8; findings are associations, not causal effects.')
if __name__=='__main__': main()
