# Kiểm toán trích xuất đặc trưng

**Trạng thái: COMPLETE cho định nghĩa và sử dụng development; PARTIAL cho provenance TEST lịch sử.** Nguồn chuẩn là `src/evaluation/phase2_inputs.py::frozen_features`; producer lịch sử tương ứng là `experiments/run_candidate_logging.py::main`. Thứ tự bắt buộc được khóa trong `src/pooladapt/selector.py::FROZEN_FEATURES` và khớp `results/phase2/checkpoints/pooladapt.metadata.json`.

| Đặc trưng | Logic đã xác nhận (input → float) | NA / tiền xử lý | Dùng selector? / leakage |
|---|---|---|---|
| `query_length` | `len(tokenize(query))`; tokenizer BM25 lower-case + `\w+` | không NA; raw rồi median-impute/scale | Có; chỉ query, không label. |
| `num_numeric_tokens` | `sum(t.isnumeric() for t in tokens)` | không NA | Có; không label. |
| `avg_idf` | mean IDF các token có trong chỉ mục BM25 | không token IDF → NaN/ô trống → TRAIN median | Có; không label. |
| `max_idf` | max IDF các token hợp lệ | như trên | Có; không label. |
| `top1_top2_rrf_margin` | `rrf_score[0]-rrf_score[1]` | pool phải đúng 100 nên hữu hạn | Có; pre-reranking. |
| `top20_overlap` | `|BM25[:20] ∩ Dense[:20]|/20` | hữu hạn | Có; pre-reranking. |
| `rank_correlation_union` | Pearson corr. hai rank, chỉ doc hiện diện cả Top-100 | <2 cặp/variance 0 → NaN | Có; imputer; không qrel. |
| `score_entropy` | `-sum(p log p)`, `p=rrf_score/sum` | pool score dương theo implementation | Có; pre-reranking. |
| `score_gini` | `2*dot(1..100,sort(score))/(100*sum)-101/100` | source logging trả 0 nếu tổng 0; frozen function giả định tổng dương | Có. |
| `redundancy` | mean Jaccard token-set title+text của mọi cặp trong RRF Top-20 | không có cặp hợp lệ → NaN | Có; không label. |
| `concentration` | `sum(rrf_score[:10])/sum(rrf_score)` | tổng 0 → NaN trong logging | Có. |
| `sparse_dense_agreement` | `1 - mean(|bm25_rank-dense_rank|)/99` trên doc chung | không doc chung → NaN | Có. |
| `sparse_dense_disagreement` | `mean(|bm25_rank-dense_rank|)/99` trên doc chung | như trên | Có. |

`src/evaluation/phase2_inputs.py::validate_pool` yêu cầu đúng 100 doc ID duy nhất và cấm trường có `qrel`, `relevance`, `m_star`, `label`. `src/pooladapt/selector.py::validate_feature_names` còn cấm các token outcome/ID. Vì vậy **mã xác nhận** không có relevance-label leakage trong vector inference; M* chỉ là target training.

Nhóm thực tế là 4 query + 9 pool = 13, đúng yêu cầu. `entity_count` có trong CSV logging nhưng trống hoàn toàn (809 NA) và không thuộc `FROZEN_FEATURES`, nên không được selector dùng. `src/pooladapt/features.py::characterize_candidates` chỉ thêm candidate-level agreement/rank-gap; không phải producer của 13 feature.

Artifact: `results/phase2/01_candidate_logging/{pool_features.csv,query_features.csv,metadata.json}` có 809 development query và ghi `qrels_read_by_feature_extractor: false`. `improvement/improvement_1/query_group_analysis/features/development_feature_consistency.md` xác nhận 3.900 cặp feature TEST tái dựng khớp hàm chuẩn, nhưng chúng **không chứng minh** score/rank thành phần lịch sử vì Dense revision lịch sử không xác minh.
