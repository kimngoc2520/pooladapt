# Kiểm toán Oracle M*

**Trạng thái: COMPLETE.** `src/analysis/oracle_mstar.py::compute_mstar_labels` tính với `BUDGETS=(10,20,30,50,100)` và `TAUS=(.95,.98,.99)`: chọn K nhỏ nhất sao cho `nDCG@10(prefix_K) >= tau * nDCG@10(full_rerank)`; nếu không có thì 100. Nó bắt buộc mọi ranking/qrel query-ID khớp chính xác. `experiments/oracle_mstar.py` ghép TRAIN+VALIDATION và từ chối result có split TEST.

| Tau | Artifact | N | M*=10/20/30/50/100 |
|---:|---|---:|---|
| .95 | `results/phase2/02_oracle/mstar_analysis_tau095.json` | 809 | 757 / 33 / 7 / 11 / 1 |
| .98 | `.../mstar_analysis_tau098.json` | 809 | 757 / 33 / 7 / 11 / 1 |
| .99 | `.../mstar_analysis_tau099.json` | 809 | 757 / 33 / 7 / 11 / 1 |

Tỷ lệ .99: 93.572% / 4.079% / 0.865% / 1.360% / 0.124%; `G1_passes=true`. Việc ba tau cho cùng labels là **artifact đã quan sát**, không phải lỗi được suy luận; cần kiểm tra sensitivity mới nếu CT2 đặt câu hỏi về tau.

Quy tắc Easy/Hard tại `experiments/analyze_easy_hard_features.py`: Easy khi M*=10, Hard khi M*>10; artifact `results/phase2/05_easy_hard_analysis/feature_easy_hard_analysis.csv` là 757 Easy và 52 Hard. Đây là nhãn **outcome-derived** từ qrels + fixed-prefix/full rerank, độc lập với prediction selector nhưng không độc lập với kết quả reranking. Không được dùng chúng làm feature hay nhóm TEST sau hoc. Định nghĩa nhất quán: checkpoint PoolAdapt cũng khóa tau=.99 và đúng rule trên.
