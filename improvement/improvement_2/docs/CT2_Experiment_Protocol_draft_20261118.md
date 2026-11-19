# PoolAdapt – Improvement 2 (CT2): Quy trình thực nghiệm

**Trạng thái:** Bản dự thảo, chờ nhóm review và phê duyệt; chưa frozen  
**Ngày cập nhật:** 18/11/2026  
**Phạm vi:** CT2.0–CT2.3 bắt buộc; CT2.4 có điều kiện; CT2.5 tùy chọn

Đây là nguồn quy trình chính, độc lập và đầy đủ cho Improvement 2 (CT2). Audit và artifact ở Section 0 là bằng chứng để xác minh hiện trạng, không phải protocol thay thế. Mọi quyết định chưa thể kiểm chứng được đánh dấu **Pending verification** hoặc **Pending group approval**. Không bắt đầu stage phụ thuộc quyết định pending cho đến khi giải quyết và ghi vào phiên bản được nhóm duyệt.

## 0. Tài liệu tham chiếu và nguyên tắc

Bằng chứng hỗ trợ (không tự phê duyệt thiết kế mới): audit/pre_execution/README.md; audit/01_feature_extraction_audit.md; audit/02_dataset_split_and_provenance_audit.md; audit/03_oracle_mstar_audit.md; audit/04_selector_implementation_audit.md; audit/05_existing_experimental_artifacts.md; audit/06_research_gaps_and_recommendations.md; audit/audit_summary.json. Mã cần đối chiếu: src/pooladapt/selector.py, experiments/run_pooladapt.py, src/evaluation/phase2.py, src/evaluation/phase2_execution.py, src/evaluation/phase2_inputs.py, src/analysis/oracle_mstar.py và artifact được audit dẫn chiếu.

Khi tài liệu mâu thuẫn với code/artifact hoặc không thể xác minh, dừng phần chạy bị ảnh hưởng, ghi nguồn/mâu thuẫn và đánh dấu Pending verification. Thay đổi protocol cần rationale, quyết định nhóm, protocol version mới và mapping run chịu ảnh hưởng; không sửa ngầm quyết định đã đăng ký.

## 1. Câu hỏi nghiên cứu và phạm vi

**RQ:** How much do candidate-pool characteristics contribute to adaptive reranking budget allocation, and what limitations prevent the current controller from achieving a better quality–cost trade-off?

- CT2.0: oracle headroom, Oracle-M* vs Fixed-10 (Gate A).
- CT2.1: controlled ablation Full/Query-only/Pool-only và Current vs cost-matched null (Gate C).
- CT2.2: controller/Hard/allocation diagnosis từ development OOF.
- CT2.3: phân tích feature và liên hệ evidence CT2.1/2.2.
- CT2.3-B: tổng hợp và quyết định có xem xét Gate E hay không.
- CT2.4: conditional improvement, chỉ sau Gate E và đăng ký/phê duyệt.
- CT2.5: cross-domain tùy chọn, không điều kiện hoàn tất CT2.

CT2.1 tạo OOF cần cho CT2.2/2.3; CT2.2/2.3 cung cấp bằng chứng cho CT2.3-B; CT2.4 phụ thuộc quyết định CT2.3-B và hồ sơ ứng viên được duyệt. Gate A/C không thay Gate E. Kết quả âm tính hoặc inconclusive là hợp lệ; không đổi RQ hay chọn contrast sau khi xem kết quả.

## 2. Dữ liệu, split, candidate pool, nhãn và feature

### 2.1 Dữ liệu và reranking
- SciFact; TRAIN=647, VALIDATION=162, TEST=300 query; ID không overlap. Development=TRAIN+VALIDATION=809.
- Candidate pool N=100, document ID duy nhất/query. Reranking dùng RRF prefix-K.
- K ∈ {10,20,30,50,100}; K=100 là Full rerank.
- Quality metrics: nDCG@10, Recall@10, MRR@10. Trong src/evaluation/retrieval_metrics.py, DCG@10=Σ(r=1..10) rel_r/log2(r+1) với gain tuyến tính; IDCG@10 sắp relevance của toàn bộ qrels query giảm dần rồi áp dụng cùng discount. nDCG=DCG/IDCG; nếu IDCG=0 thì trả 0. Recall@10 là số relevant ID duy nhất trong prefix-10 chia tổng relevant ID; nếu không có relevant ID thì 0. MRR@10 là nghịch đảo rank relevant đầu tiên trong top 10, hoặc 0 nếu không có. Dùng implementation tại commit chạy; ghi hash/version, không đổi implementation giữa methods.

### 2.2 M* và Hard
τ=0.99. M*(q) là K nhỏ nhất sao cho nDCG@10(prefix-K,q) ≥ 0.99 × nDCG@10(full,q), hoặc 100 nếu không K nào đạt. Hard=(M*>10). Audit xác nhận development có 757 M*=10 và 52 Hard/809. Các τ=.95/.98/.99 có cùng phân bố trong artifact hiện tại; đây là quan sát, không phải căn cứ đổi τ. M* là outcome-derived label, không phải feature/model input.

### 2.3 Feature
Thứ tự bắt buộc theo src/pooladapt/selector.py::FROZEN_FEATURES:
1. top1_top2_rrf_margin
2. top20_overlap
3. rank_correlation_union
4. score_entropy
5. score_gini
6. redundancy
7. concentration
8. sparse_dense_agreement
9. sparse_dense_disagreement
10. query_length
11. num_numeric_tokens
12. avg_idf
13. max_idf

Pool-only là 1–9; Query-only là 10–13; Full có 13 cột lưu. Agreement/disagreement là biểu diễn bổ sung của cùng tín hiệu, không phải hai tín hiệu độc lập. entity_count trống trên 809 development query và bị loại. Feature tính trước Cross-Encoder; không dùng qrels, relevance, M*, Hard, IDs hoặc outcome sau reranking làm input. Missing values median-impute. TEST feature tái dựng phải gọi reconstructed/not historical; không chứng minh feature lịch sử.

## 3. Controller hiện tại và baseline

### 3.1 Implementation hiện có
experiments/run_pooladapt.py tạo Pipeline: SimpleImputer(median), StandardScaler, LogisticRegression(max_iter=1000, class_weight=balanced, random_state=42). Checkpoint sidecar xác nhận lbfgs, C=1, tol=1e-4, fit_intercept=True. Fit hiện có trên TRAIN=647, preprocessing chỉ fit TRAIN; không resampling. Feature order Section 2.3; classes={10,20,30,50,100}; TRAIN class counts lần lượt 605/23/7/11/1.

Selector ánh xạ predict_proba theo model.classes_, lấy log(max(p_k,1e-12)); lớp không hiện diện được clamp 1e-12, đây là fallback số học. g(K)=−(K−mean(G))/std(G), với population standard deviation. z’_k(q)=z_k(q)/T+λg(k). λ>0 ưu tiên K nhỏ; λ<0 ưu tiên K lớn. Argmax; tie chọn K nhỏ. Chọn RRF prefix-K rồi Cross-Encoder rerank; không chọn candidate riêng lẻ.

### 3.2 Calibration implementation hiện có
src/evaluation/phase2.py::calibrate chỉ dùng logits/ID VALIDATION, kiểm tra population và cấm TRAIN/TEST. Một λ chung cho mọi query, T đã cho (mặc định 1). Quét các vùng policy λ trên toàn trục thực qua giao điểm logit; Average-K chỉ từ budget prediction, không dùng ranking outcome. Implementation targets {10,20,30,50}; chọn Average-K gần target nhất, tie thì |λ| nhỏ hơn rồi λ nhỏ hơn. Mỗi target có relative error = |Average-K−target|/target; implementation ghi SUCCESS khi relative error ≤5% **và** policy responsive. Trong implementation, responsive được xác định bằng sanity check ở λ={−5,0,5}: tính Average-K trên VALIDATION tại cả ba giá trị; responsive=true nếu các giá trị Average-K không đồng nhất (ít nhất hai giá trị khác nhau), false nếu cả ba giống nhau. Đây là phép đo variation của predicted workload khi lambda thay đổi, không phải ngưỡng độ lớn do protocol đặt thêm. Nếu false, implementation chỉ xét λ=0 và đánh dấu target FAILED dù relative error ≤5%; phải lưu ba Average-K, responsive=false, trạng thái FAILED và không gọi đó là calibration success. Class absent được clamp ở selector; calibrator không có mask riêng. Đây là calibration validation-only của implementation, không phải pooled-development OOF. run_pooladapt.py đánh dấu calibration của nó là sanity-only, không final matched-budget calibration.

### 3.3 Calibration và baselines CT2
Đây là hành vi của implementation hiện có, không tự động là procedure CT2 controlled-CV. Controlled-CV dùng Development folds; không mặc định VALIDATION là một calibration set riêng và cũng không mặc định gộp OOF. Pending group approval trước CT2.1: calibration population; T cố định/tìm; cách aggregate prediction; lambda search/tie-break; λ shared across queries; xử lý absent class (mask hay clamp); Average-K targets/tolerance; unavailable target handling. Calibration implementation hiện có dùng λ chung cho mọi query, exhaustive policy-region sweep ở T cố định và tie-break gần λ=0 rồi λ nhỏ hơn; CT2 chưa được duyệt áp dụng procedure đó cho từng fold/OOF. Không dùng quality để match cost trừ khi duyệt trước.

- Fixed Prefix-K: đúng RRF prefix K.
- Current: Full 13-feature, 5-class LR + lambda-shift.
- Cost-matched null: giữ budget distribution nhưng phá liên hệ query feature-budget. Cách randomization/permutation, seed, repeats, expected quality, target/tolerance và inference là Pending verification.
- Oracle-M*: outcome-aware, optimistic reference, không khả triển khai.
- Cost-matched oracle: công thức/procedure Pending verification; chưa chạy.

Gate A=Oracle-M* vs Fixed-10; Gate C=Current vs null; CT2.4=Candidate vs Current. Không trộn lẫn.

## 4. Trình tự và phụ thuộc

1. CT2.0/Gate A: development reranking, qrels, M* → paired quality/CI/headroom.
2. CT2.1: frozen dev features/labels, prefix scores, qrels → ba ablations, OOF và Gate C nếu null được duyệt. Giải quyết pending trước confirmatory run.
3. CT2.2: Full Current OOF, M*, quality/cost → diagnosis.
4. CT2.3: dev features, folds/OOF và CT2.1/2.2 → contribution analysis.
5. CT2.3-B/Gate E: tổng hợp headroom, Gate C và diagnosis; ghi activate/no activate/inconclusive.
6. CT2.4: chỉ sau Gate E và amendment được duyệt.
7. CT2.5: optional; nếu kích hoạt, scope và quyết định phải được đăng ký bằng dated amendment trong tài liệu này theo Section 10.
8. Final audit: provenance, manifests, hashes, deviations, limitations.

Không bỏ qua/gộp gate để có kết quả thuận lợi.

## 5. CT2.0 – Oracle headroom (Gate A)

Trên development: d_q=nDCG@10(M*(q),q)−nDCG@10(10,q); Δ_A=mean_q(d_q). Giữ δ_A=0.005 như giá trị nêu trong protocol hiện tại; group cần xác nhận khi duyệt freeze vì audit không xác minh phê duyệt lịch sử.

Báo N, định nghĩa nDCG, phân bố d_q, mean paired difference và CI 95%. Implementation trả 0 khi IDCG=0. Phương pháp CI và seed/count resampling là Pending group approval; không bắt đầu phân tích Gate A để quyết định trước khi đăng ký. Khi được duyệt: CI lower bound >δ_A là headroom vượt ngưỡng; upper bound <δ_A là dưới ngưỡng; CI chứa δ_A là inconclusive. Không-significant không có nghĩa tương đương. M* outcome-derived/optimistic; Gate A không chứng minh current selector tốt hoặc pool feature hữu ích. Không TEST.

## 6. CT2.1 – Controlled feature ablation

### 6.1 Config và CV
- Full=13; Query-only=4; Pool-only=9.
- Development=809; không TEST.
- Repeated Stratified 5-fold CV × 10 repeats, seeds 0–9; strata {10}, {20,30}, {50,100}; inner Stratified 3-fold trên outer train.
- LR C grid {0.01,0.1,1,10}; imputer/scaler fit trong training fold; cấu hình dùng cùng outer/inner folds, seeds, evaluation queries.
- Không xem repeated predictions là query độc lập. Cách aggregate repeated probabilities/logits và calibration sau aggregate là Pending group approval.
- Lưu class counts/mask khi absent; không giả rằng class absent đã học. Alignment và aggregation absent class Pending group approval.
- Audit, implementation và artifact đã kiểm tra không xác nhận một inner-OOF objective/selection rule được nhóm duyệt. Outer metrics không tự động trở thành inner objective. Exact objective metric, cách aggregate scores qua inner folds, quy tắc chọn C và tie-break C đều **Pending group approval**; không suy đoán từ outer endpoints. CT2.1 confirmatory CV không được bắt đầu cho đến khi objective và tie-break được nhóm duyệt và ghi vào protocol amendment/version.

### 6.2 Contrasts và inference
Primary Full−Query-only. Secondary Full−Pool-only và Pool-only−Query-only. Quality: nDCG@10, Recall@10, MRR@10; cost: predicted/actual K, mean pairs, compression=1−mean(K)/100; latency báo riêng. Paired CI, tests, correction family, alpha và metric hierarchy Pending group approval. Không dùng nhãn chung “permutation/bootstrap” khi chưa đăng ký statistic, unit, count/seed và correction.

### 6.3 Gate C
δ_C=0.0025, Average-K targets {11,12,14,17,20}, tolerance ±0.1 được ghi trong bản hiện tại nhưng chưa audit/source xác minh độc lập: proposed values, Pending group approval. Đây là proposed CT2 cost-matching grid, khác mục đích với implementation operating-point targets {10,20,30,50}. ±0.1 là absolute Average-K tolerance: |observed Average-K−target|≤0.1; nó không phải relative error. Implementation tolerance 5% dùng |observed Average-K−target|/target≤0.05. Không thay target set này bằng target set kia và chưa dùng proposed CT2 values trước group approval. Target nào không match được sẽ báo unavailable; chính sách giữ/bỏ target khi aggregate endpoints Pending group approval.

Null phải bảo toàn budget distribution và phá mapping query-feature-budget. Nếu được duyệt, nDCG_null(q)=Σ_k p_k·nDCG(q,k); cách lấy p_k, expectation vs permutation, count/seed/inference Pending verification/group approval. Frontier dự kiến d_q(c)=nDCG(q,K_q)−nDCG_null(q), Δ(c)=mean_q d_q(c); common cost levels, aggregation, paired bootstrap và Gate C decision Pending group approval. Gate C chỉ Current-vs-null.

## 7. CT2.2 – Controller diagnosis

Positive class là Hard=1[M*>10]; prevalence development hiện có 52/809≈0.0643. Với năm-class model, score Hard là P(Hard)=P(K=20)+P(K=30)+P(K=50)+P(K=100), ánh xạ theo class labels chứ không theo vị trí cột cố định.

**Metric implementation đã tồn tại:** experiments/run_multivariate_feasibility.py dùng sklearn.metrics.average_precision_score(y_true, y_score), với Hard=1 là positive và predict_proba[:,1] làm score trong binary Easy/Hard feasibility experiment. Đây là Average Precision (AP), sklearn's non-interpolated step-wise summary: Σ_n (R_n−R_{n−1})P_n trên các threshold/rank điểm; đây không phải trapezoidal auc(recall, precision), và hai giá trị không được gọi thay thế cho nhau. Script hiện có tính AP riêng cho mỗi outer test fold của RepeatedStratifiedKFold (5 folds × 10 repeats), rồi báo mean/std trên các fold; nó không tạo một aggregated OOF score/query. Audit phân loại kết quả này là binary feasibility, không phải CT2 5-class controller diagnosis.

**CT2 controlled-CV metric:** implementation/method cho query-level aggregation repeated OOF scores chưa được duyệt; lựa chọn này và việc áp dụng average_precision_score lên vector query-level sau aggregation là **Pending group approval**. Không tính/báo CT2 confirmatory PR-AUC từ repeated fold rows như thể độc lập, cũng không tự chuyển sang trapezoidal PR-AUC. Khi procedure được duyệt, ghi function/version, y_true, aggregated y_score, positive label=1, và cùng-population Hard prevalence làm no-skill AP reference. AP baseline prevalence là reference, không phải hiệu quả mô hình.

Absent five-class model classes: existing selector maps model.classes_ to budget grid and assigns 1e-12 before log to any missing class; it does not mask/renormalize that class. Controlled-CV fold missing-class handling and score aggregation remain Pending group approval, per Sections 3.3/6.1; do not fabricate probabilities for absent classes. Hard precision/recall/F1/confusion matrix require a discrete decision rule; that rule for CT2 diagnosis is also Pending group approval unless fixed before the run. Report per-class/macro/support where meaningful, subgroup N and rare-class uncertainty. Allocation errors: signed K_pred−M*, absolute |K_pred−M*|, ordinal class error; under-allocation K_pred<M* and over-allocation K_pred>M* separately. Report quality loss/cost. Generate one OOF vector/query only after approved aggregation; repeated folds are not independent queries. TEST is excluded from Hard detection, allocation diagnosis and method selection.

## 8. CT2.3 – Feature contribution

Mục tiêu descriptive/predictive, không causal. Dùng development/folds/OOF; không chọn feature/threshold/model bằng TEST. Audit/code hỗ trợ: phân bố/missingness, association với M*/Hard/allocation (gọi là association), query-vs-pool group, tương quan/multicollinearity, tính bổ sung agreement/disagreement, liên hệ ablation và controller failure modes; báo N/subgroup counts.

Permutation importance trên held-out folds, drop-group ngoài ablation, fold stability, association tests/correction là Pending group approval; không bắt buộc nếu chưa duyệt. Correlated/complementary features có thể chia sẻ/làm thấp permutation importance. Không suy ra causality; có thể kết luận chưa đủ evidence.

## 9. CT2.4 – Conditional improvement

### 9.1 Gate E
Gate E criteria/ngưỡng và bằng chứng CT2.2/2.3 cụ thể Pending group approval. CT2.3-B phải ghi diagnosis được support/refute, Gate A, Gate C, limitations và lý do candidate xử lý limitation. Gate A/C không thay Gate E và không là candidate acceptance threshold.

Không bắt đầu CT2.4 trước khi nhóm duyệt: hypothesis; method/formulation; search grid/tie-break; population/folds/seeds/calibration; primary endpoint/family; cost matching/tolerance/unavailable targets; tests/CI/multiplicity; δ_imp nếu có; decision criteria; artifacts. Tất cả Pending group approval. Không tự đặt hyperparameters/grid/δ_imp/alpha/acceptance.

### 9.2 Hypotheses (categories, chưa duyệt)
- H1: ordinal/cost-sensitive Logistic Regression khi có evidence mismatch lớp/chi phí; formulation cần đăng ký.
- H2: GBDT/Random Forest chỉ khi CT2.3 hỗ trợ nonlinear interactions; model/grid cần đăng ký.
- H3: threshold policy trên P(Hard), threshold chỉ chọn từ development.
- H4: continuous quality-loss/cost-sensitive objective nếu M* rời rạc thiếu thông tin; objective phải đăng ký.

Đây không phải search plan. Feature mới cần hypothesis/leakage review. Không mở rộng grid sau khi xem kết quả hoặc chỉ báo candidate thắng.

### 9.3 Baseline, endpoints, report
Current=Full 13-feature five-class LR + lambda shift trên folds giống candidate. Candidate-vs-current khác Gate A/C. Targets {11,12,14,17,20} và tolerance ±0.1 Pending group approval. Matching chỉ theo budget distribution nếu được duyệt, không dùng M*/Hard/quality. Unavailable target, common-level aggregation, primary/secondary endpoints, tests, CIs, multiplicity, δ_imp và criteria đều pending. Không gọi “đạt” nếu chưa có decision rule. Lưu mọi registered candidate, OOF, matching, metrics, inference, environment, decision memo; không dùng TEST tuning/calibration/Gate E/selection.

## 10. CT2.5 – Cross-domain tùy chọn

CT2.5 không phải prerequisite hoàn tất CT2. Không cần protocol document riêng. Trước bất kỳ CT2.5 experiment nào, nhóm phải duyệt proposed scope và đăng ký: research question; target domain; dataset và provenance/split; candidate-pool construction/feature availability; transfer hay retraining; calibration; metrics; baselines; statistical methods/CI/multiplicity; acceptance criteria; TEST boundary; và required artifacts. Không tự chọn domain/dataset hoặc suy ra tiêu chí chấp nhận.

Sau khi duyệt, ghi các quyết định thành dated amendment ngay trong CT2_Experiment_Protocol.md. Amendment phải nêu ngày, protocol version trước/sau, SHA-256 của protocol được duyệt, approval information (nhóm/người phê duyệt và xác nhận), cùng scope/design đã đăng ký. Những mục chưa quyết định phải ghi Pending group approval/verification; CT2.5 không bắt đầu khi bất kỳ quyết định thiết yếu nào còn pending. Không dùng CT2.5 để thay kết quả CT2 âm tính.

## 11. TEST và provenance

**Existing implementation:** TRAIN=647 là population dùng fit selector hiện có và median imputer/StandardScaler. VALIDATION=162 là population riêng dùng cho calibration λ-only đã mô tả ở Section 3.2. TEST=300 bị loại khỏi fit và calibration. Đây là vai trò trong implementation hiện tại.

**Controlled-CV experiments:** Development gồm TRAIN+VALIDATION=809; outer/inner folds phải theo CV design được duyệt, không gộp 809 query để fit một model rồi gọi dự đoán trên chính query đó là OOF. Mỗi preprocessing object chỉ fit trên training fold tương ứng. Mỗi query's OOF prediction phải được tạo khi label query đó không tham gia model fitting. Calibration phải theo procedure controlled-CV riêng đã phê duyệt; VALIDATION không mặc định là standalone calibration set trong mọi fold, và cũng không mặc định dùng pooled OOF. TEST bị loại khỏi training, preprocessing fit, calibration, feature/hyperparameter/policy selection. TEST=300 chỉ final evaluation với config khóa; không dùng cho threshold, Gate A/C/E, CT2.4 activation hay Hard/allocation diagnosis.

Audit ghi 2 FAIL và 3 NOT VERIFIED ở provenance checks. Hai FAIL liên quan evaluation-script hash và source-manifest/amendment hash không khớp lịch sử; chưa verify historical TEST feature/group, run ID/timestamp riêng, expected frozen-output digest. Giữ caveat trong reports. TEST reconstructed features không phải historical; RRF match không xác minh Dense component/features. Nếu final TEST được phép, báo Average-K thực tế/workload mismatch; không tuning/cost-match trên TEST.

## 12. Reproducibility và artifacts

Mỗi run lưu: protocol version/SHA-256; repo commit/working tree; dataset/split/query và fold/inner-fold manifests/hash; feature order/input hashes; command/config/seeds/runtime/dependencies/hardware/timestamp; model/checkpoint hashes; class counts; preprocessing fit population; calibration T/λ/population/search/tie-break; query-level predictions/metric inputs/budget; metrics/CI/tests/sample sizes; deviations.

Stage artifacts: CT2.0 paired query data/summary; CT2.1 per-fold/OOF và ablation/null/Gate C outputs; CT2.2 diagnosis; CT2.3 analyses; CT2.4 hypothesis/config register, all candidates and decision memo; final adherence/provenance audit/index. Không ghi đè kể cả failed/negative/superseded. Tên/schema mới chưa quy định: khai báo convention/schema version, run ID, stage trong manifest trước run; không nhận là yêu cầu lịch sử.

## 13. Thay đổi protocol

Không tự đổi split, N, K, τ, feature/order, label, CV/fold/seed, C grid, contrast, metric, tests/correction, calibration, cost targets/tolerance hoặc gates. Giá trị proposed/pending không mặc định đã khóa. Thay đổi cần rationale, affected sections, group decision, version, timestamp và run mapping.

## 14. Checklist trước khi chạy

- [ ] Group review/approval, version/hash.
- [ ] Commit/working tree, split IDs/no overlap, N=100, unique candidate IDs.
- [ ] Feature order đúng, entity_count bỏ, không leakage.
- [ ] TEST cách ly và provenance caveat trong report.
- [ ] Inner objective/C tie-break, absent class, OOF aggregation, calibration đã giải quyết.
- [ ] Metrics/CI/tests/multiplicity, Gate A/C procedures đã duyệt theo stage.
- [ ] Null/target/tolerance duyệt nếu Gate C.
- [ ] CT2.4: Gate E và hypothesis/method/grid/endpoints/matching/criteria duyệt.
- [ ] Nếu kích hoạt CT2.5: dated amendment trong protocol này được nhóm duyệt, gồm scope, version/hash và approval record.
- [ ] Run folder/schema/command/config/seeds/environment/hashes sẵn; không ghi đè.
- [ ] Blocker resolved hoặc stage dừng.

Không tích pending bằng suy đoán; checklist stage phải đạt trước khi bắt đầu stage.

## 15. Freeze và versioning

Tài liệu là draft cho đến khi group review/approval; không tự frozen. Freeze yêu cầu: (1) phê duyệt nhóm; (2) commit protocol; (3) SHA-256; (4) repo commit reference; (5) freeze timestamp; (6) mapping runs với protocol version/hash. Sửa sau freeze tạo version mới, rationale và run mapping; giữ lịch sử thay đổi.

Trình tự: hoàn tất checklist stage → xác minh input/manifests → chạy Section 4 → lưu mọi kết quả/deviation → final audit.
