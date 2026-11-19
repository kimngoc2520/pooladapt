# Kiểm toán artifact thí nghiệm CT2 hiện có

Phân loại dựa trên file hiện có; COMPLETE không đồng nghĩa đáp ứng controlled ablation.

| Hạng mục yêu cầu | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| Full-feature selector | COMPLETE | `results/phase2/10_pooladapt/pooladapt_results.csv`, checkpoint và final TEST `evaluation/final_gpu/*`. |
| Query-only selector | MISSING | Không tìm thấy checkpoint/result có 4 query feature riêng. |
| Pool-only selector | MISSING | Không tìm thấy checkpoint/result có 9 pool feature riêng. |
| Feature coefficients | COMPLETE | `06_multivariate_selector/feature_coefficients.csv`; là classifier Easy/Hard feasibility, không phải 5-class selector attribution. |
| Feature–M* association | COMPLETE | `03_analysis/feature_mstar_analysis_tau095/098/099.csv`, Spearman/Holm. |
| Easy/Hard distributions | COMPLETE | `05_easy_hard_analysis/feature_easy_hard_analysis.csv`, 757/52; development only. |
| ROC-AUC / PR-AUC | COMPLETE (PARTIAL scope) | `06_multivariate_selector/multivariate_results.json`: balanced full-feature 0.6126 / 0.1268 repeated CV; binary Easy/Hard, không TEST/5-class. |
| Hard Precision/Recall | MISSING | Không có report hard-class precision/recall cho PoolAdapt allocation. |
| Confusion matrix | MISSING | Không có artifact confusion matrix selector-vs-oracle. |
| Budget allocation error / Average K | COMPLETE | `evaluation/final_gpu/{comparative_results.csv,quality_cost_latency.csv}`; PoolAdapt K 10.267,20.833,32.867,49.333. B30 unmatched. |
| Query-level quality/latency | COMPLETE | `evaluation/final_gpu/test_query_metrics.csv` và `test_predictions.jsonl`, 300 TEST IDs; latency reranking-only. |
| Full TEST comparative quality-cost-latency | COMPLETE | `quality_cost_latency.csv`, `phase2_summary.json`, Full/RR/Fixed/PACE/SAGE/PoolAdapt. |
| Historical TEST feature vector provenance | NOT VERIFIED | `improvement/improvement_1/audit/test_provenance_audit_report.md` FEATURE-01; reconstructed features không là historical proof. |

Các số final TEST PoolAdapt (nDCG@10; mean latency giây) tại K target 10/20/30/50: 0.67485;0.05724 — 0.67041;0.11616 — 0.67560;0.17204 — 0.68158;0.24892. Đây là kết quả thực nghiệm đã lưu, không suy ra causal benefit và không thay thế ablation.
