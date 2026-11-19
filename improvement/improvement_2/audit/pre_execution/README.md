# Kiểm toán kho mã cho Improvement 2 (CT2 – Method)

Ngày kiểm toán: 2026-11-17. Mục tiêu là xác định chính xác hiện trạng PoolAdapt trước CT2; không chạy lại thí nghiệm, không sửa mã nguồn hay kết quả sẵn có.

## Phạm vi và phương pháp

Đã đọc mã nguồn, khoá split SciFact, checkpoint/sidecar, các CSV/JSON Phase 2 và kết quả TEST cuối. Mỗi kết luận được gắn một trong bốn mức: **đã xác nhận bằng mã**, **được hỗ trợ bởi artifact**, **suy luận/chưa xác minh**, hoặc **thiếu**. Các artifact TEST tái dựng của Improvement 1 được phân biệt với artifact Phase 2 lịch sử.

| Tệp | Nội dung |
|---|---|
| `01_feature_extraction_audit.md` | 13 đặc trưng, công thức, NA/leakage và mức sử dụng. |
| `02_dataset_split_and_provenance_audit.md` | split, ID, provenance và rủi ro TEST. |
| `03_oracle_mstar_audit.md` | M*, ngưỡng, Easy/Hard và phân bố lớp. |
| `04_selector_implementation_audit.md` | Logistic Regression, pipeline, suy luận và checkpoint. |
| `05_existing_experimental_artifacts.md` | Ma trận COMPLETE/PARTIAL/MISSING/NOT VERIFIED cho bằng chứng CT2. |
| `06_research_gaps_and_recommendations.md` | Các khoảng trống và thí nghiệm CT2 còn cần làm. |
| `audit_summary.json` | Tóm tắt máy đọc được. |

## Trạng thái tổng quan

**Có thể chuẩn bị CT2, nhưng chưa thể coi là đã hoàn tất Feature Ablation.** Toàn bộ 13 đặc trưng và selector 5 lớp đã được xác minh trên tập phát triển. Split SciFact là 647 TRAIN / 162 VALIDATION / 300 TEST, không chồng lấp. M* ở cả ba ngưỡng tồn tại nhưng cho cùng phân bố đã lưu. Final TEST có quality, pair-count và latency; tuy nhiên provenance hiện hành còn hai FAIL và ba NOT VERIFIED theo `improvement/improvement_1/audit/test_provenance_audit_report.md`.

Kết quả đa biến, hệ số và Easy/Hard hiện có là phân tích feasibility/phát hiện liên hệ, **không** là ablation kiểm soát query-only/pool-only/full trên protocol CT2.

## Bước tiếp theo

Khóa protocol CT2 trước khi chạy: so sánh full vs query-only vs pool-only trên cùng split/seed/M*; fit imputer/scaler chỉ ở TRAIN; chọn calibration chỉ VALIDATION; đánh giá TEST đúng một lần; báo ROC/PR, hard Precision/Recall, confusion matrix, sai số ngân sách, K, chất lượng và latency theo query.
