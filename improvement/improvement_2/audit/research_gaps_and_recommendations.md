# Khoảng trống nghiên cứu và khuyến nghị CT2

## Khoảng trống đã xác nhận

1. **Thiếu ablation kiểm soát.** Không có query-only/pool-only selector, nên không thể kết luận pool-aware feature mang lợi ích vượt query-only hay full model.
2. **Thiếu đánh giá allocation trực tiếp.** Cần confusion matrix giữa `predicted K` và oracle M*, hard-query Precision/Recall, MAE/signed error K và under/over-allocation rate.
3. **Feasibility không là kết quả CT2.** `06_multivariate_selector` dự đoán Hard binary bằng repeated CV; hệ số/ROC/PR của nó không đo allocation 5 mức và không là controlled feature ablation.
4. **Provenance TEST cần được trình bày thận trọng.** Hai FAIL và ba NOT VERIFIED ở Improvement 1 audit ngăn tuyên bố tái lập hoàn toàn; feature TEST tái dựng không thể thay artifact lịch sử.
5. **Mất cân bằng target cực mạnh.** TRAIN .99 có 605/647 ở K=10 và chỉ một K=100; cần macro/per-class report và hard-focused analysis, không chỉ accuracy/average K.

## Protocol CT2 khuyến nghị (chưa thực hiện)

- Giữ 647/162/300 split, candidate pool, M* tau=.99, seed và policy grid cố định; chỉ thay subset feature.
- Chạy ba pipeline cùng preprocessing nội bộ TRAIN: full 13, query-only 4, pool-only 9. Lưu feature order, fitted artifact hashes và class counts.
- Chọn/calibrate lambda chỉ trên VALIDATION; khóa một lần trước TEST. Không dùng TEST để feature selection, threshold hay model choice.
- Báo cáo trên TEST theo cùng target budget: nDCG/Recall/MRR, actual pairs, average K, P50/P95 reranking latency, paired CIs/tests; đánh dấu workload mismatch như B30 hiện tại.
- Thêm macro/per-class precision/recall/F1, confusion matrix, hard(M*>10) precision/recall, allocation signed/absolute error, và query-level paired deltas. Báo cáo association/coefficients như mô tả, không như causal importance.
- Nếu cần kiểm tra tau, lập sensitivity .95/.98/.99 bằng các labels có sẵn trước; không tạo/hậu chọn protocol bằng TEST.

Không khuyến nghị chạy Feature Ablation trong nhiệm vụ này; đây chỉ là danh sách công việc còn thiếu.
