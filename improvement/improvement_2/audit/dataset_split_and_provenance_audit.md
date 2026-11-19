# Kiểm toán split và provenance dữ liệu

**Trạng thái: COMPLETE cho split/ID hiện có; PARTIAL cho chuỗi provenance thực thi TEST lịch sử.**

| Split | Số query | Bằng chứng |
|---|---:|---|
| TRAIN | 647 | `data/scifact/splits/train_query_ids.txt` và `split_metadata.json` |
| VALIDATION | 162 | `data/scifact/splits/validation_query_ids.txt` |
| TEST | 300 | ID duy nhất trong `data/scifact/qrels/test.tsv`; cũng được khóa tại `results/phase2/evaluation/locked_configurations.json` |

Đối chiếu trực tiếp ID cho kết quả overlap TRAIN–VAL=0, TRAIN–TEST=0, VAL–TEST=0. `src/data/scifact_phase2.py::split_original_train` tạo 80/20 từ ID original TRAIN đã sort, `random.Random(42)`, sau đó `validate_split` bắt buộc partition đầy đủ và không chạm TEST. Hai qrels con được tạo bởi `_write_filtered_qrels`.

Training tại `experiments/run_pooladapt.py::main` chỉ gọi `model.fit` với 647 TRAIN rows. Imputer và StandardScaler nằm trong sklearn `Pipeline`, do đó chỉ fit TRAIN. `src/evaluation/phase2.py::calibrate` kiểm tra population chỉ VALIDATION và cấm TRAIN/TEST; `results/phase2/evaluation/provenance_amendment.json` ghi `test_used_for_calibration: false`. TEST chỉ được dùng sau selection để tính metric.

Provenance development có manifest/hash: `results/phase2/01_candidate_logging/metadata.json`, hash input sidecar checkpoint và `results/phase2/checkpoints/checkpoint_manifest.json`. Checkpoint ghi SHA-256, runtime versions và reload check 16 TRAIN rows PASS. TEST final có `final_gpu/execution_manifest.json`, prediction/metric/summary, nhưng Improvement 1 audit ghi hai FAIL: hash script evaluation hiện tại khác execution manifest; amendment/source-manifest hiện tại khác hash được manifest lịch sử tham chiếu. Ba mục NOT VERIFIED: feature/group TEST gốc, run ID/timestamp riêng, và expected frozen-output digest.

Artifact tái dựng TEST dưới `improvement/improvement_1/query_group_analysis/` là **reconstructed**, dù Gate A xác nhận 300 RRF ordering khớp frozen pools. Theo `features/feature_provenance.md`, Dense revision/raw component scores lịch sử không được xác minh; không được gọi là historical feature provenance.
