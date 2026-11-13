# PoolAdapt: Xếp hạng lại có chọn lọc dựa trên đặc tính tập ứng viên quy mô lớn trong Agentic Hybrid RAG dưới ràng buộc chi phí suy luận

*Phân tích nhóm truy vấn bổ sung cho Improvement 1 · tập TEST SciFact được cố định (300 truy vấn)*

> Đây là phân tích mô tả được thực hiện sau, dựa trên các đặc trưng truy xuất đã tái dựng. Báo cáo này cập nhật trạng thái “chưa thể thực hiện” về phân tích nhóm truy vấn trong báo cáo Improvement 1 trước đó. Hai trường `controller_features` và `query_group` trong handoff gốc vẫn để trống; các nhóm dưới đây là tầng phân tích, không phải đầu ra bộ điều khiển hay một chính sách mới đã được đánh giá.

## 1. Tóm tắt điều hành

PoolAdapt nghiên cứu liệu tín hiệu của tập ứng viên có thể quyết định số lượng ứng viên được đưa qua Cross-Encoder hay không. Phân tích nhóm truy vấn xem xét liệu các truy vấn TEST được cố định có khác nhau theo cách liên quan đến quyết định này. Đây là phân tích bổ sung cho Improvement 1, không phải một đóng góp thuật toán độc lập.

Phân tích dùng 300 truy vấn TEST SciFact, các danh sách Top-100 BM25, Dense và RRF đã lưu, đặc trưng trước xếp hạng lại được tái dựng, cùng kết quả theo truy vấn đã cố định của PoolAdapt và Fixed Prefix. Danh sách RRF tái dựng khớp chính xác thứ tự và tập ứng viên của pool Phase 2 đã lưu trên cả 300 truy vấn. Hai phép chia trung vị cân bằng, độc lập với kết quả đánh giá được khảo sát: tính dư thừa từ vựng Top-20 và mức giao nhau Top-20 giữa BM25 và Dense.

Kết quả nổi bật là nhóm có độ giao tập ứng viên cao có chất lượng truy xuất RRF trước xếp hạng lại tốt hơn nhóm có độ giao tập ứng viên thấp: nDCG@10 trung bình lần lượt là 0.7308 và 0.6325; Recall@10 là 0.8651 và 0.7467. Ở nhãn ngân sách mục tiêu B30, nDCG@10 của PoolAdapt là 0.7399 trong nhóm có độ giao tập ứng viên cao và 0.6113 trong nhóm có độ giao tập ứng viên thấp; chênh lệch ghép cặp so với Fixed Prefix lần lượt là +0.0179 và −0.0027. Đây là đối chiếu mô tả, không phải tương tác nhóm–phương pháp đã được kiểm định. Số cặp thực tế cũng khác nhãn ngân sách, nên B30 không phải so sánh có chi phí bằng nhau.

Bằng chứng cho thấy điều kiện truy xuất không đồng nhất và gợi ý rằng đặc tính tập ứng viên có thể giúp nhận diện nhu cầu xếp hạng lại khác nhau. Kết quả không chứng minh chính sách thích ứng cải thiện chất lượng hoặc chi phí theo quan hệ nhân quả. Thứ hạng và điểm Dense được tái dựng bằng phiên bản mô hình chưa xác nhận là phiên bản lịch sử; việc khớp thứ tự RRF chỉ xác nhận thứ tự pool hợp nhất, không xác minh thứ hạng BM25/Dense thành phần, điểm thô hay phiên bản Dense lịch sử.

## 2. Động cơ nghiên cứu

Nếu mọi truy vấn đều được xếp hạng lại trên toàn bộ pool lớn, Cross-Encoder phải xử lý cả những truy vấn mà danh sách ứng viên hợp nhất có thể đã được sắp xếp tốt. Ngược lại, nếu mọi truy vấn đều bị giới hạn cùng một mức, hệ thống có thể bỏ qua bằng chứng hữu ích khi truy xuất thưa và dày không đồng thuận. Câu hỏi của PoolAdapt là liệu đặc tính quan sát được của pool có thể hướng dẫn ngân sách xếp hạng lại theo truy vấn trong điều kiện chi phí suy luận bị giới hạn mà vẫn duy trì chất lượng hay không.

Phân tích nhóm giúp kiểm tra câu hỏi này ở mức truy vấn. Nó cho thấy liệu chất lượng truy xuất, kết quả xếp hạng lại và chi phí đo được có khác nhau giữa các điều kiện pool trước xếp hạng lại hay không. Những đối chiếu này có thể gợi mở cho lựa chọn dựa trên đặc tính tập ứng viên, nhưng không chứng minh đặc trưng gây ra kết quả hoặc bộ chọn có thể khai thác đặc trưng ổn định.

## 3. Câu hỏi nghiên cứu

- **RQ1 — Mức biến thiên đặc trưng:** Tính dư thừa của pool, mức giao nhau ứng viên, đồng thuận thứ hạng, phân bố điểm và mô tả truy vấn thay đổi thế nào? **Có thể trả lời ở mức mô tả**, với lưu ý về Dense tái dựng.
- **RQ2 — Tính bổ sung giữa truy xuất thưa và dày:** Danh sách Top-*k* của BM25 và Dense giao nhau đến đâu, thứ hạng của tài liệu chung giống nhau thế nào? **Có thể trả lời với thứ hạng thành phần tái dựng**; thứ tự RRF khớp không chứng minh thứ hạng BM25/Dense lịch sử.
- **RQ3 — Khác biệt giữa các nhóm:** Các nhóm được gán trước có khác nhau về chất lượng truy xuất và kết quả PoolAdapt so với Fixed Prefix không? **Có thể mô tả**, nhưng chưa có kiểm định tương tác xác nhận.
- **RQ4 — Chất lượng và chi phí:** nDCG@10, số cặp được xếp hạng lại và độ trễ thay đổi thế nào theo nhóm? **Có thể trả lời tại các điểm vận hành đã lưu**; độ trễ chỉ thuộc giai đoạn xếp hạng lại và nhãn ngân sách không bảo đảm chi phí thực tế bằng nhau trong từng nhóm.
- **RQ5 — Hàm ý chính sách:** Những quan sát này có xác nhận giá trị của bộ chọn thích ứng không? **Không.** Chúng tạo động cơ cho thí nghiệm tiếp theo nhưng không xác thực chính sách mới trên TEST.

## 4. Dữ liệu và phương pháp thực nghiệm

### Tập dữ liệu cố định và nguồn gốc

Tập TEST gồm 300 ID duy nhất trong `results/phase2/evaluation/locked_configurations.json`. Các đầu ra phân nhóm bao phủ đúng các ID này. Mỗi bảng BM25, Dense và RRF có 30.000 dòng: 100 thứ hạng duy nhất, liên tục cho mỗi truy vấn; không có cặp truy vấn–tài liệu hay truy vấn–thứ hạng trùng. RRF tái dựng đạt `GATE_A_EXACT_MATCH`: thứ tự và tập Top-100 khớp 300/300 pool Phase 2 đã cố định. Điều này xác nhận thứ tự ứng viên hợp nhất, không xác nhận điểm BM25/Dense lịch sử hay phiên bản Dense lịch sử.

Dense Retrieval được tái dựng cục bộ trên CPU bằng `sentence-transformers/all-MiniLM-L6-v2`; phiên bản được gắn nhãn `RECONSTRUCTED_CANDIDATE_REVISION_NOT_VERIFIED_HISTORICAL`. Không khẳng định điểm thành phần tái dựng giống số học với lần chạy GPU gốc. Điểm BM25, Dense và RRF được giữ riêng, không so sánh độ lớn giữa các thang điểm khác nhau.

### Đặc trưng và tái dựng

Bảng đặc trưng có một dòng cho mỗi truy vấn TEST, 109 cột, không có ô thiếu và không có `query_id` trùng. 13 đặc trưng gốc của PoolAdapt được tính lại và khớp hàm chuẩn `frozen_features` trên cả 3.900 cặp truy vấn–đặc trưng tại `rtol=1e-10`, `atol=1e-12`. Giá trị TEST được tính vẫn là giá trị tái dựng. Bảng 809 dòng TRAIN/VALIDATION chỉ dùng để đối chiếu lược đồ và phân bố; tập này không giao với TEST và không thay cho bất kỳ giá trị TEST nào.

13 đặc trưng thuộc bốn nhóm có ý nghĩa với câu hỏi nghiên cứu:

1. **Mức dứt khoát của RRF:** khoảng cách điểm giữa hạng 1 và 2, entropy điểm đã chuẩn hóa, Gini điểm và tỷ trọng điểm Top-10 mô tả mức tập trung của điểm RRF tái dựng. Chúng có thể gợi ý liệu tiền tố ngắn đã đủ dứt khoát hay chưa, nhưng phụ thuộc điểm tái dựng.
2. **Tính bổ sung thưa–dày:** độ giao nhau Top-20, tương quan thứ hạng BM25–Dense trên tài liệu chung Top-100 và độ dịch chuyển thứ hạng chuẩn hóa/đồng thuận suy ra mô tả liệu hai tín hiệu có trả về cùng ứng viên và xếp các ứng viên chung tương tự hay không. Giao nhau và đồng thuận thứ hạng là hai thuộc tính khác nhau.
3. **Tính dư thừa văn bản:** Jaccard tập token trung bình giữa tài liệu RRF Top-20 đo mức lặp lại từ vựng. Đây là đại diện từ vựng; nó không xác định trùng lặp ngữ nghĩa hay độ bao phủ bằng chứng.
4. **Mô tả truy vấn:** số token, số token dạng số và IDF BM25 trung bình/cao nhất mô tả độ dài và độ đặc hiệu từ vựng. Các yếu tố này có thể ảnh hưởng truy xuất nhưng không dùng để chia hai nhóm được báo cáo.

Kích thước pool và thống kê vị trí thứ hạng là hằng số vì mỗi truy vấn có 100 ứng viên; chúng không phân biệt truy vấn trong phân tích này. Tỷ lệ tài liệu rỗng và nội dung trùng chính xác cũng bằng không.

### Phân nhóm và đánh giá

Nhóm được gán trước khi nối kết quả đánh giá, chỉ dựa trên trung vị đặc trưng; giá trị bằng ngưỡng được xếp vào nhóm thấp. Nhóm dư thừa dùng Jaccard tập token trung bình giữa tài liệu RRF Top-20. Nhóm giao thưa–dày dùng **Jaccard tập ứng viên Top-20** từ danh sách BM25 và Dense tái dựng; cột nội bộ `sparse_dense_agreement_group` được giữ vì tương thích; nó biểu thị Jaccard tập ứng viên, không phải thứ hạng. Mỗi phép chia có 150 truy vấn thấp và 150 truy vấn cao. Qrels, nhãn liên quan và kết quả xếp hạng lại không dùng để gán nhóm. Kết quả TEST chỉ được nối sau đó bằng `query_id` chính xác.

Chất lượng truy xuất trước xếp hạng lại được tính từ thứ hạng Top-100 đã lưu và qrels TEST bằng các hàm `ndcg_at_k`, `recall_at_k`, `mrr_at_k` của kho mã (nDCG dùng gain liên tục theo định nghĩa dự án). Đây là phép tính chỉ đọc; không chạy lại truy xuất, xếp hạng lại hay suy luận mô hình. So sánh sau xếp hạng lại dùng `test_query_metrics.csv` đã cố định, ghép cặp theo truy vấn, cho PoolAdapt và Fixed Prefix ở nhãn ngân sách 10, 20, 30 và 50. Bảng hiện có báo cáo trung bình, trung vị, độ lệch chuẩn, chênh lệch ghép cặp và khoảng bootstrap phân vị theo truy vấn (2.000 lần lặp, seed 42). Các khoảng này mang tính thăm dò, không hiệu chỉnh cho nhiều so sánh nhóm/chỉ số/ngân sách và không phải kiểm định xác nhận ý nghĩa thống kê.

Chỉ số chất lượng gồm nDCG@10, Recall@10 và MRR@10. Chi phí được thể hiện bằng số cặp thực tế được xếp hạng lại trên mỗi truy vấn và độ trễ đo được. Bộ định giờ bao gồm chuẩn bị cặp, suy luận Cross-Encoder và sắp xếp; không gồm truy xuất, lựa chọn, nạp mô hình và làm nóng. Đây không phải độ trễ đầu cuối hay chi phí tiền tệ/GPU.

## 5. Kết quả và phân tích

### 5.1. Phân tích đặc trưng truy vấn

Các thống kê dưới đây là trung bình hoặc trung vị theo truy vấn trên 300 truy vấn. Hai phép đo giao nhau dùng mẫu số khác nhau: `top20_overlap` gốc của PoolAdapt bằng giao/20; đặc trưng phân nhóm là Jaccard, giao/hợp.

| Nhóm đặc trưng / phép đo | Thống kê quan sát | Diễn giải và giới hạn |
|---|---:|---|
| Kích thước tập ứng viên | 100 cho mọi truy vấn | Cố định theo thiết kế; không thể tạo khác biệt phân bổ giữa truy vấn. |
| Giao BM25–Dense Top-20 | Trung bình 6.02/20; `top20_overlap` trung bình 0.301 | Hai bộ truy xuất đóng góp nhiều ứng viên khác nhau ở đầu danh sách. |
| Jaccard tập Top-20 | Trung bình 0.190; trung vị 0.160 | Có biến thiên giữa truy vấn; trung vị tách nhóm giao thấp và cao. |
| Giao tập Top-100 | Jaccard trung bình 0.165; trung vị 0.149; trung bình 27.26 tài liệu chung | Phần lớn tài liệu trong mỗi danh sách nguồn không trùng nhau; RRF cắt hợp các danh sách về Top-100. |
| Tương quan hạng và dịch chuyển trên tài liệu chung | Tương quan hạng Pearson trung bình 0.278; dịch chuyển tuyệt đối trung bình 26.25 hạng; đồng thuận chuẩn hóa suy ra trung bình 0.735 | Giao tập không mô tả đầy đủ mức đồng thuận thứ tự. Cột phụ `component_shared_rank_spearman` được tính bằng `numpy.corrcoef`, tức tương quan Pearson; báo cáo theo phép tính thay vì tên cột. |
| Rank-Biased Overlap, p=0.90 | Trung bình 0.355 | Mức trùng khớp thứ tự toàn danh sách vừa đến thấp trong lần tái dựng này. |
| Dư thừa từ vựng RRF Top-20 | Jaccard trung bình 0.1085; trung vị 0.1069 | Tài liệu Top-20 có phần token chung hạn chế; phân bố đủ biến thiên cho một phép chia mô tả. |
| Nội dung trùng chính xác / tài liệu rỗng | 0 trong Top-20 và Top-100 | Không thấy trùng token-set chính xác hay tài liệu rỗng; không loại trừ dư thừa ngữ nghĩa. |
| Tỷ lệ token duy nhất | Trung bình 0.524 trong Top-20; 0.310 trong Top-100 | Khi mở rộng pool, phần token hợp duy nhất nhỏ hơn tương đối so với tổng token của từng tài liệu. Đây là đại diện từ vựng cho tính đa dạng. |
| Điểm RRF tái dựng | Entropy trung bình 4.531 (tối đa ln(100)=4.605); Gini 0.206; khối lượng điểm Top-10 0.191; khoảng cách hạng 1–2 là 0.00145 | Gợi ý phân bố điểm tương đối trải rộng, nhưng không phải điểm lịch sử đã xác minh. |
| Mô tả truy vấn | Trung bình 13.07 token; 0.213 token dạng số; IDF trung bình theo truy vấn 3.401 và IDF cao nhất 6.877 | Độ dài và IDF biến thiên nhưng không tạo nhóm báo cáo; IDF phụ thuộc chỉ mục BM25 tái dựng. |

Các phân bố cho thấy kích thước pool cố định không đồng nghĩa cấu trúc pool đồng nhất. Mức giao ứng viên, thứ tự và tính dư thừa từ vựng thay đổi dù mọi truy vấn đều có 100 ứng viên hợp nhất. Đây là tín hiệu khả dĩ cho phân bổ ngân sách, nhưng chưa chứng minh giá trị dự báo đối với lợi ích xếp hạng lại.

### 5.2. Phân bố nhóm truy vấn

| Chiến lược phân nhóm | Trung bình / trung vị nhóm thấp | Ngưỡng | Trung bình / trung vị nhóm cao | Số truy vấn |
|---|---:|---:|---:|---:|
| Dư thừa từ vựng Top-20 | 0.0977 / 0.0990 | 0.106945 | 0.1194 / 0.1160 | 150 / 150 |
| Jaccard giao tập ứng viên Top-20 BM25–Dense | 0.0914 / 0.1111 | 0.159664 | 0.2890 / 0.2500 | 150 / 150 |

Hai phép chia cân bằng và có thể diễn giải như các tầng mô tả. Ngưỡng trung vị phụ thuộc mẫu TEST, làm mất biến thiên liên tục và không phải ngưỡng vận hành. Nhóm giao tập ứng viên phụ thuộc các thứ hạng thành phần BM25/Dense tái dựng nên kế thừa giới hạn nguồn gốc.

### 5.3. Chất lượng truy xuất trước xếp hạng lại

| Danh sách đã lưu | nDCG@10 trung bình | Recall@10 trung bình | MRR@10 trung bình |
|---|---:|---:|---:|
| BM25 | 0.6519 | 0.7740 | 0.6186 |
| Dense | 0.6451 | 0.7833 | 0.6047 |
| RRF | 0.6816 | 0.8059 | 0.6472 |

Trong các danh sách đã lưu/tái dựng, RRF có trung bình cao nhất trên cả ba chỉ số. Đây là so sánh mô tả trên 300 truy vấn TEST. Thứ tự ứng viên RRF khớp pool cố định, nhưng phiên bản Dense lịch sử chưa xác minh; không nên trình bày kết quả thành phần như so sánh truy xuất lịch sử đã được xác nhận.

Chất lượng RRF cũng khác nhau giữa các nhóm:

| Nhóm | Số truy vấn | nDCG@10 RRF trung bình (ĐLC) | Recall@10 trung bình | MRR@10 trung bình |
|---|---:|---:|---:|---:|
| Dư thừa thấp | 150 | 0.6679 (0.4082) | 0.7833 | 0.6332 |
| Dư thừa cao | 150 | 0.6954 (0.3669) | 0.8284 | 0.6611 |
| Nhóm có độ giao tập ứng viên thấp | 150 | 0.6325 (0.4233) | 0.7467 | 0.6017 |
| Nhóm có độ giao tập ứng viên cao | 150 | 0.7308 (0.3429) | 0.8651 | 0.6926 |

Trung vị Recall@10 và MRR@10 đều bằng 1.0 ở cả bốn nhóm, nên trung bình và phân bố hữu ích hơn trung vị đối với các chỉ số dễ chạm trần. Nhóm có độ giao tập ứng viên cao bắt đầu với chất lượng RRF tốt hơn; chênh lệch sau xếp hạng lại cần được diễn giải theo đường cơ sở này, không xem mức giao tập ứng viên tự nó là nguyên nhân tạo lợi ích.

### 5.4. Tương quan thứ hạng, giao tập ứng viên, tính dư thừa và đa dạng

Jaccard Top-20 BM25–Dense trung bình là 0.190 (trung vị 0.160), còn Jaccard Top-100 trung bình là 0.165. Ở Top-100, hai danh sách chia sẻ trung bình khoảng 27/100 tài liệu. Tương quan thứ hạng trên tài liệu chung thấp hơn mức đồng thuận hoàn toàn (Pearson trung bình 0.278), trong khi mức đồng thuận dịch chuyển chuẩn hóa có trung bình 0.735. Hai phép đo trả lời câu hỏi khác nhau: một đo dịch chuyển hạng trên tài liệu chung; phép còn lại đo giao tập ứng viên ở độ sâu xác định. Không phép đo nào tự thân cho biết Cross-Encoder sẽ thay đổi thứ tự Top-10 tốt nhất ra sao.

Tính dư thừa từ vựng Top-20 trung bình là 0.1085, nhưng tỷ lệ nội dung trùng chính xác bằng không. Kết quả cho thấy mức giao từ vựng biến thiên; không chứng minh nhóm dư thừa cao chứa bằng chứng trùng lặp hay có thể bỏ qua xếp hạng lại an toàn. Không có chỉ số đa dạng ngữ nghĩa hoặc độ bao phủ bằng chứng.

### 5.5. Xếp hạng lại và đánh đổi chất lượng–chi phí theo nhóm

Bảng trình bày so sánh tại nhãn ngân sách mục tiêu B30. Nhãn này khác với số cặp thực tế được xếp hạng lại và độ trễ đo được trong giai đoạn xếp hạng lại. RRF là chất lượng trước xếp hạng lại; PoolAdapt và Fixed Prefix lấy từ kết quả theo truy vấn đã cố định. Chênh lệch ghép cặp bằng PoolAdapt trừ Fixed Prefix; khoảng là bootstrap thăm dò 2.000 lần. Thay đổi tương đối nDCG tính trên trung bình Fixed Prefix trong nhóm.

| Nhóm | nDCG@10 RRF | nDCG@10 Fixed Prefix | nDCG@10 PoolAdapt | Δ nDCG@10 (KTC 95%; tương đối) | Cặp TB, PoolAdapt / Fixed | Độ trễ TB (ms), PoolAdapt / Fixed |
|---|---:|---:|---:|---:|---:|---:|
| Dư thừa cao | 0.6954 | 0.6870 | 0.6961 | +0.0091 [−0.0060, 0.0266]; +1.32% | 37.07 / 30.00 | 193.49 / 175.84 |
| Dư thừa thấp | 0.6679 | 0.6490 | 0.6551 | +0.0060 [−0.0161, 0.0284]; +0.93% | 28.67 / 30.00 | 150.60 / 174.80 |
| Nhóm có độ giao tập ứng viên cao | 0.7308 | 0.7221 | 0.7399 | +0.0179 [0.0022, 0.0369]; +2.47% | 31.80 / 30.00 | 169.62 / 175.88 |
| Nhóm có độ giao tập ứng viên thấp | 0.6325 | 0.6140 | 0.6113 | −0.0027 [−0.0256, 0.0203]; −0.45% | 33.93 / 30.00 | 174.47 / 174.76 |

Khoảng B30 của nhóm có độ giao tập ứng viên cao nằm trên 0 trong bảng thăm dò chưa hiệu chỉnh, nhưng không có hiệu chỉnh so sánh nhiều lần hoặc kiểm định tương tác nhóm–phương pháp; không xem đây là tuyên bố ý nghĩa thống kê xác nhận. Ở B30, PoolAdapt dùng trung bình 31.8 cặp trong nhóm có độ giao tập ứng viên cao và 33.93 trong nhóm có độ giao tập ứng viên thấp, so với 30 của Fixed Prefix. Cùng nhãn mục tiêu không tạo chi phí thực tế bằng nhau giữa các nhóm.

Ở B50, trong nhóm có độ giao tập ứng viên cao, PoolAdapt dùng 45.27 cặp so với 50 của Fixed Prefix (−9.47%) và độ trễ trung bình 234.07 ms so với 257.38 ms; chênh lệch nDCG@10 là +0.00268 (KTC 95% [0.00000, 0.00781]). Trong nhóm có độ giao tập ứng viên thấp, PoolAdapt dùng 53.40 cặp (+6.80%) và độ trễ 263.78 ms so với 250.70 ms; chênh lệch nDCG@10 là +0.00142 (KTC 95% [−0.00850, 0.01042]). Đây là đối chiếu mô tả theo nhóm về khối lượng và kết quả, không chứng minh mức giao tập ứng viên dự báo lợi ích nhân quả.

Phân tích tổng thể Improvement 1 xác định B10, B20 và B50 là các so sánh chính theo ngân sách cặp thực tế; B30 là bổ sung vì trung bình số cặp lệch quá ngưỡng ghép ±5%. Do đó, so sánh nhóm tại B30 càng không phù hợp để xem là so sánh có chi phí bằng nhau. Báo cáo tổng thể Improvement 1 ghi nhận không có giá trị p hoán vị đã hiệu chỉnh Holm nào dưới 0.05 trong các kiểm định chính.

Các tệp đã kiểm tra không có hình riêng cho Query-group Analysis. Hình Improvement 1 hiện có tổng hợp phương pháp, không tách nhóm; báo cáo dùng bảng đã xác minh và không gắn nhãn nhầm các hình đó thành phân tích nhóm.

## 6. Thảo luận

Các nhóm phản ánh điều kiện truy xuất khác nhau. Truy vấn thuộc nhóm có độ giao tập ứng viên cao có chất lượng RRF trước xếp hạng lại tốt hơn nhóm có độ giao tập ứng viên thấp; nhóm có độ giao tập ứng viên thấp cũng có độ biến thiên nDCG lớn hơn. Điều này gợi ý tư cách nhóm liên quan đến độ khó truy vấn hoặc chất lượng bằng chứng truy xuất, nhưng không tách riêng tác động nhân quả của xếp hạng lại.

Kết quả xếp hạng lại thay đổi theo nhóm và ngân sách. Ở B30, chênh lệch PoolAdapt so với Fixed Prefix dương trong nhóm có độ giao tập ứng viên cao và hơi âm trong nhóm có độ giao tập ứng viên thấp; số cặp thực tế và độ trễ của PoolAdapt thay đổi trái chiều giữa hai nhóm dư thừa. Ở B50, nhóm có độ giao tập ứng viên cao có số cặp và độ trễ quan sát được thấp hơn với PoolAdapt, còn nhóm có độ giao tập ứng viên thấp có giá trị cao hơn. Mẫu hình phù hợp với giả thuyết nhu cầu xếp hạng lại không đồng nhất, nhưng khoảng tin cậy, nhiều phép so sánh, cách lập nhóm và chênh lệch ngân sách hạn chế sức mạnh diễn giải. Các thứ hạng BM25/Dense dùng để tạo nhóm giao là kết quả tái dựng. Việc khớp thứ tự pool RRF chỉ xác nhận pool hợp nhất, không xác minh thứ hạng BM25/Dense lịch sử, điểm thô hay phiên bản Dense.

Hàm ý thực tiễn cho PoolAdapt là xem đặc trưng pool như tín hiệu ứng viên cho bộ chọn được hiệu chỉnh trên TRAIN/VALIDATION, không phải quy tắc TEST cố định. Kiểm định chính sách hợp lệ cần cố định đặc trưng, mô hình và hiệu chỉnh trước TEST; so sánh tại ngân sách cặp thực tế tương đương; và đánh giá chất lượng, độ trễ trên tập mới chưa sử dụng. Phân tích nhóm này không xác thực chính sách thích ứng.

## 7. Hạn chế và nguy cơ đối với tính hợp lệ

- **Quần thể:** 300 truy vấn TEST SciFact, chia thành nhóm 150. Kết quả có thể không khái quát sang FiQA, HotpotQA, corpus hoặc hệ Agentic Hybrid RAG khác.
- **Tái dựng:** Thứ tự RRF khớp không xác minh thứ hạng BM25/Dense lịch sử, điểm thô hay phiên bản Dense lịch sử. Đặc trưng phụ thuộc thành phần tái dựng còn mang tính sơ bộ.
- **Tên đặc trưng:** Trường phụ `component_shared_rank_spearman` được tính bằng tương quan Pearson. Đầu vào mô hình gốc `rank_correlation_union` cũng là Pearson theo hàm chuẩn. Báo cáo công khai điểm này, không âm thầm sửa tên hoặc công thức.
- **Phân nhóm:** Chia trung vị nhạy với mẫu TEST, làm mất thông tin liên tục và không phải ngưỡng của chính sách học được. Hai chiều phân nhóm được phân tích riêng; chưa phân tích giao của chúng.
- **Phạm vi chỉ số:** Recall@10 dùng tổng tài liệu liên quan theo qrels. Handoff xếp hạng lại không có Recall@100. Độ trễ bỏ qua truy xuất, lựa chọn, nạp mô hình và làm nóng; không phải độ trễ đầu cuối hay chi phí tiền tệ.
- **Suy luận:** Khoảng bootstrap mang tính thăm dò, không hiệu chỉnh trên toàn bộ so sánh nhóm. Không thực hiện kiểm định tương tác chính thức; chênh lệch trung bình không chứng minh nhân quả.
- **Khả dụng vận hành:** Các cột `controller_features` và `query_group` trong handoff Improvement 1 gốc vẫn trống. Nhóm phân tích được tính riêng, không ghi vào handoff.
- **Đa dạng:** Jaccard tập token và tỷ lệ token duy nhất là đại diện từ vựng; riêng các đặc trưng này không đo dư thừa ngữ nghĩa, độ đủ bằng chứng hay mức liên quan tài liệu.

## 8. Kết luận và hàm ý cho PoolAdapt

Pool RRF cố định ở 100 ứng viên cho mọi truy vấn nhưng đặc tính trước xếp hạng lại vẫn biến thiên. Tính dư thừa Top-20 và mức giao tập ứng viên giữa truy xuất thưa và dày tạo thành các nhóm mô tả cân bằng, độc lập kết quả. Nhóm giao tập ứng viên cho thấy chênh lệch lớn về chất lượng truy xuất RRF trước xếp hạng lại. Kết quả PoolAdapt so với Fixed Prefix thay đổi theo nhóm và điểm vận hành, gồm các mẫu hình tương phản về số cặp và độ trễ ở B50.

Các kết quả ủng hộ tiền đề rằng điều kiện truy vấn và pool có thể quan trọng khi phân bổ công sức xếp hạng lại. Thứ hạng BM25/Dense dùng để xác định mức giao là kết quả tái dựng; kiểm tra thứ tự RRF không xác minh thứ hạng thành phần lịch sử hay phiên bản Dense. Chúng không chứng minh hiệu quả chính sách thích ứng, không xác lập cơ chế nhân quả và không giải quyết nguồn gốc Dense lịch sử. Improvement 1 vẫn là thí nghiệm chính; Query-group Analysis bổ sung góc nhìn mô tả và giả thuyết cho một đánh giá chính sách được hiệu chỉnh riêng.

## 9. Khả năng tái lập và tham chiếu hiện vật

- **Đầu vào:** ID cố định tại `results/phase2/evaluation/locked_configurations.json`; qrels TEST tại `data/scifact/qrels/test.tsv`; kết quả xếp hạng lại tại `results/phase2/evaluation/final_gpu/test_query_metrics.csv`; ID ứng viên đã lưu trong `test_predictions.jsonl`.
- **Bảng truy xuất và đặc trưng:** [BM25](../reconstruction/bm25_test_top100.csv), [Dense](../reconstruction/dense_test_top100.csv), [RRF](../reconstruction/rrf_test_top100.csv), [đặc trưng truy vấn](../features/test_query_features.csv), [ma trận 13 đặc trưng](../features/original_13_feature_matrix_reconstructed.csv) và [nhãn nhóm](../groups/query_group_assignments.csv).
- **Kết quả:** [chất lượng trước xếp hạng lại theo nhóm](pre_reranking_retrieval_quality_by_group.csv) và [so sánh PoolAdapt–Fixed Prefix](subgroup_comparison.csv). Bảng sau có bốn nhãn ngân sách, năm chỉ số, trung bình/trung vị/độ lệch chuẩn theo nhóm và khoảng bootstrap ghép cặp.
- **Phương pháp và kiểm định:** [phương pháp phân nhóm](../groups/grouping_methodology.md), [nguồn gốc đặc trưng](../features/feature_provenance.md), [kiểm định nhất quán đặc trưng](../features/development_feature_consistency.md), [kiểm định RRF](../reconstruction/rrf_pool_validation_summary.md), [kiểm toán toàn vẹn dữ liệu](../audit/data_integrity_audit.md), [kiểm toán rò rỉ](../audit/leakage_audit.md) và [báo cáo xác thực cuối](../audit/final_validation_report.md).
- **Mã nguồn:** `reconstruct_test_retrieval.py`, `extract_test_features.py`, `assign_query_groups.py`, `evaluate_query_groups.py` và script bổ sung `evaluate_retrieval_quality.py` trong `../scripts/`. Lệnh bổ sung là `py -3.11 improvement/improvement_1/query_group_analysis/scripts/evaluate_retrieval_quality.py`; script đọc thứ hạng và qrels đã lưu, đồng thời từ chối ghi đè đầu ra. Chỉ chạy trong cây đầu ra mới hoặc sau khi đã lưu giữ bản tổng hợp hiện có. Các script tái dựng và phân nhóm cũng bảo vệ đầu ra đã tồn tại; không chạy lại pipeline đã hoàn tất chỉ để đọc báo cáo.
- **Kiểm tra tái lập:** đã chạy kiểm tra cú pháp/import và test tập trung sau khi sửa báo cáo và phân tích; không chạy lại truy xuất, suy luận Cross-Encoder, huấn luyện hay toàn bộ pipeline thí nghiệm để tạo báo cáo này.

`reconstruction/artifact_checksums.json` ghi các đầu ra tái dựng ban đầu và có trước bảng chất lượng bổ sung cùng hai báo cáo này. Tệp đó không bị ghi đè; bảng phân tích mới có thể tái tạo bằng script và các đầu vào cố định được dẫn ở trên.
