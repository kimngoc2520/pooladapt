"""Create a self-contained final results canvas from completed frozen artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/phase2/07_evaluation'


def render() -> Path:
    """Embed real validation/TEST data with distinct tables and numeric plots."""
    summary = json.loads((ROOT / 'results/phase2/phase2_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'COMPLETE':
        raise ValueError('cannot render final TEST findings before evaluation completes')
    data = {'calibration': [r for r in summary['configurations'] if r['method'] in ('PoolAdapt', 'SAGE-SLO')],
            'results': summary['final_test_results'], 'statistics': summary['statistical_results'],
            'gates': summary['gates'], 'drift': summary['test_budget_drift'],
            'historical_audit': {name: {'mismatches': info['historical_prediction_mismatches'], 'historical_sanity': info['historical_sanity_average_K'],
                                       'frozen_sanity': info['frozen_checkpoint_sanity_average_K']} for name, info in summary['historical_reproduction_audit'].items()}}
    source = TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':'), allow_nan=False))
    path = OUT / 'Phase2-results.canvas.tsx'
    path.write_text(source, encoding='utf-8')
    print(path)
    return path


TEMPLATE = r'''import { H1, H2, Text, Table, Stack, Grid, Select, useState, useHostTheme } from "cursor/canvas";
const data = __DATA__;
const names = ["Full Rerank", "Fixed Prefix", "PACE-EF", "Random", "SAGE-SLO", "PoolAdapt"];
const fmt = (v: number | null, digits=4) => v === null ? "—" : v.toFixed(digits);
const pct = (v: number | null) => v === null ? "—" : (100*v).toFixed(1)+"%";
type Point = {method:string; target_budget:number; actual_average_budget:number|null; [key:string]:any};
function Plot({title, x, y, xlabel, ylabel}: {title:string;x:string;y:string;xlabel:string;ylabel:string}) {
 const t=useHostTheme();
 const rows=data.results.filter((r:Point)=>r[x]!==null&&r[y]!==null);
 const maxX=Math.max(...rows.map((r:Point)=>Number(r[x])))*1.08;
 const maxY=y==="latency_reduction"?1:1;
 const minY=y==="latency_reduction"?Math.min(0,...rows.map((r:Point)=>Number(r[y]))):0;
 const px=(v:number)=>64+v/maxX*580;
 const py=(v:number)=>260-(v-minY)/(maxY-minY)*216;
 const stroke=(name:string)=>name==="PoolAdapt"?t.accent.primary:t.text.secondary;
 const dash=["", "", "5 3", "2 3", "9 3", ""];
 return <section><H2>{title}</H2><svg viewBox="0 0 720 310" role="img" aria-label={title} style={{width:"100%",color:t.text.secondary}}>
  {[0,.25,.5,.75,1].map(f=><g key={f}><line x1="64" x2="644" y1={py(minY+(maxY-minY)*f)} y2={py(minY+(maxY-minY)*f)} stroke={t.stroke.tertiary}/><text x="54" y={py(minY+(maxY-minY)*f)+4} textAnchor="end" fontSize="11" fill="currentColor">{(minY+(maxY-minY)*f).toFixed(2)}</text><text x={px(maxX*f)} y="279" textAnchor="middle" fontSize="11" fill="currentColor">{(maxX*f).toFixed(x==="avg_reranked_pairs"?0:2)}</text></g>)}
  <line x1="64" x2="644" y1="260" y2="260" stroke={t.stroke.primary}/><line x1="64" x2="64" y1="44" y2="260" stroke={t.stroke.primary}/>
  <text x="354" y="301" textAnchor="middle" fontSize="12" fill="currentColor">{xlabel}</text><text transform="translate(16 152) rotate(-90)" textAnchor="middle" fontSize="12" fill="currentColor">{ylabel}</text>
  {names.map((name,i)=>{const points=rows.filter((r:Point)=>r.method===name).sort((a:Point,b:Point)=>a[x]-b[x]);return <g key={name} stroke={stroke(name)} fill={stroke(name)}>
   {points.length>1&&<polyline points={points.map((r:Point)=>`${px(r[x])},${py(r[y])}`).join(" ")} fill="none" strokeWidth={name==="PoolAdapt"?2.3:1.2} strokeDasharray={dash[i]}/>}
   {points.map((r:Point)=><g key={r.target_budget}><title>{`${name}; B=${r.target_budget}; actual K=${fmt(r.actual_average_budget,3)}; ${y}=${fmt(r[y])}; matched=${r.matched_budget}`}</title>{i%3===0?<circle cx={px(r[x])} cy={py(r[y])} r={name==="Full Rerank"?5:3}/>:i%3===1?<rect x={px(r[x])-3} y={py(r[y])-3} width="6" height="6"/>:<path d={`M${px(r[x])} ${py(r[y])-4} l4 8 h-8 z`}/>}</g>)}
  </g>;})}
 </svg><div style={{display:"flex",gap:14,flexWrap:"wrap",fontSize:12}}>{names.map((name,i)=><span key={name} style={{color:stroke(name)}}>{["●","■","▲"][i%3]} {name}{dash[i]?` (${i===2?"dashed":i===3?"dotted":"long dash"})`:""}</span>)}</div>
 <Text size="small" tone="secondary">Source: final SciFact TEST, 300 queries, 6 October 2026. Random points are five-seed means. Numeric x positions use actual measured workload or mean reranking seconds. Lines connect operating points; no Pareto frontier is inferred.</Text>
 </section>;
}
export default function Phase2Results(){
 const t=useHostTheme(); const [budget,setBudget]=useState("all");
 const rows=data.results.filter((r:Point)=>budget==="all"||r.target_budget===Number(budget)||r.method==="Full Rerank").sort((a:Point,b:Point)=>a.target_budget-b.target_budget||names.indexOf(a.method)-names.indexOf(b.method));
 const significant=data.statistics.filter(r=>r.p_Holm<.05);
 const mismatches=data.historical_audit["SAGE-SLO"].mismatches;
 return <Stack gap={18} style={{padding:24,color:t.text.primary,background:t.bg.editor}}>
 <H1>PoolAdapt · Final Phase 2</H1><Text>Frozen checkpoints: 647 TRAIN queries. Budget calibration: 162 VALIDATION queries. Final comparison: the same 300 SciFact TEST queries. Parameters and hypotheses were locked before TEST.</Text>
 <H2>A. VALIDATION CALIBRATION</H2><Text size="small">T stayed at 1. Lambda alone controlled hard argmax. Configurations were chosen by workload error, without quality-based selection.</Text>
 <Table headers={["Method","Target B","Actual mean K","Relative error","T","Lambda","Validation nDCG@10","Status"]} rows={data.calibration.map(r=>[r.method,r.target_budget,fmt(r.actual_average_budget,3),pct(r.relative_budget_error),r.temperature,fmt(r.budget_bias_lambda,6),fmt(r["validation_nDCG@10"]),r.calibration_status])}/>
 <H2>B. FINAL 300-QUERY TEST</H2><Select value={budget} onChange={setBudget} options={[{value:"all",label:"All budgets"},...[10,20,30,50].map(b=>({value:String(b),label:`B = ${b}`}))]}/>
 <Table headers={["Method","Target B","Actual mean K","nDCG@10","Recall@10","MRR@10","Compression","Mean latency (s)","P50 (s)","P95 (s)","Latency reduction","Matched ±5%"]} rows={rows.map((r:Point)=>[r.method,r.target_budget,fmt(r.actual_average_budget,3),r.method==="Random"?`${fmt(r["nDCG@10"])} ± ${fmt(r["nDCG@10_seed_std"])}`:fmt(r["nDCG@10"]),r.method==="Random"?`${fmt(r["Recall@10"])} ± ${fmt(r["Recall@10_seed_std"])}`:fmt(r["Recall@10"]),r.method==="Random"?`${fmt(r["MRR@10"])} ± ${fmt(r["MRR@10_seed_std"])}`:fmt(r["MRR@10"]),pct(r.compression_ratio),fmt(r.mean_reranking_latency),fmt(r.p50_reranking_latency),fmt(r.p95_reranking_latency),pct(r.latency_reduction),r.matched_budget?"Yes":"No"])} stickyHeader/>
 <Text size="small">Random: mean ± sample SD across seeds 0,1,2,3,4. Latency percentiles for Random are averages of seed-level percentiles. CSV latencies are seconds. Compression = 1 − mean K/100. Reranking includes pair preparation, inference, and sorting; loading, warm-up, retrieval, selection, and progress printing are excluded.</Text>
 <H2>Quality–Cost–Latency Trade-off Analysis</H2>
 <Grid columns={2}><Plot title="nDCG@10 vs average reranked pairs" x="avg_reranked_pairs" y="nDCG@10" xlabel="Average reranked document pairs / query" ylabel="Mean nDCG@10 (ratio)"/><Plot title="Recall@10 vs average reranked pairs" x="avg_reranked_pairs" y="Recall@10" xlabel="Average reranked document pairs / query" ylabel="Mean Recall@10 (ratio)"/><Plot title="MRR@10 vs average reranked pairs" x="avg_reranked_pairs" y="MRR@10" xlabel="Average reranked document pairs / query" ylabel="Mean MRR@10 (ratio)"/><Plot title="nDCG@10 vs mean reranking latency" x="mean_reranking_latency" y="nDCG@10" xlabel="Mean reranking latency (seconds / query)" ylabel="Mean nDCG@10 (ratio)"/></Grid>
 <Plot title="Latency reduction relative to Full Rerank" x="avg_reranked_pairs" y="latency_reduction" xlabel="Average reranked document pairs / query" ylabel="Mean latency reduction (ratio)"/>
 <H2>Paired statistical comparisons</H2><Text>Two-sided Wilcoxon signed-rank test; zero differences excluded; 300 paired queries. Global Holm correction across all {data.statistics.length} predeclared eligible comparisons/metrics. Random uses the per-query mean across five seeds. {significant.length} corrected comparisons are significant; p ≥ .05 means no statistically significant difference detected, not equivalence.</Text>
 <Table headers={["Method A","B","Method B","B","Metric","Mean A − B","Raw p","Holm p","Finding"]} rows={data.statistics.map(r=>[r.method_a,r.budget_a,r.method_b,r.budget_b,r.metric,fmt(r.mean_difference_a_minus_b,5),r.p_value.toPrecision(4),r.p_Holm.toPrecision(4),r.interpretation])}/>
 <H2>Protocol and limitations</H2><Text>G1 PASS · G2 FAILED · G3a NOT_EVALUATED · G3b NOT_EVALUATED. No heuristic was reconstructed. Multivariate feasibility does not replace G2. PACE-EF is Evidence Frontloading followed by fixed Prefix-K; SAGE-SLO and PoolAdapt adapt the RRF-prefix budget.</Text>
 <Text>{data.drift.length?`${data.drift.length} adaptive TEST operating points drifted outside the matched-budget tolerance. Their actual K is reported and they are excluded from matched-budget tests, without retuning.`:"All adaptive TEST operating points stayed within the matched-budget tolerance."}</Text>
 <Text>Historical SAGE development predictions differ from the frozen checkpoint for {mismatches}/162 validation queries. Checkpoint integrity, recipe, ordered features, and training population passed verification. Older outputs were preserved; their provenance does not establish the cause of this discrepancy. Final evaluation uses the explicitly frozen checkpoints.</Text>
 <Text>Scope: SciFact only; adapted baselines; no answer generation, end-to-end SLO, or retrieval/selector latency claim. PoolAdapt adapts K and keeps RRF ordering; this is not candidate-level intrinsic relevance selection. No superiority or equivalence is inferred solely from descriptive means.</Text>
 </Stack>;
}
'''


if __name__ == '__main__':
    render()
