# Báo cáo cá nhân — Data Ingestion Engineering

**Học viên:** Nguyễn Đức Nhật  
**MSSV:** 2A202603812  
**Corpus:** 5 PDF giải tích, 249 trang nội dung theo log và metrics hiện có  
**Benchmark:** 20 câu, gồm 15 dev và 5 test; giữ nguyên file benchmark và kết quả đã lưu.

## Tóm tắt thiết kế

Pipeline dùng `pypdf` để trích xuất văn bản theo trang, chuẩn hóa Unicode/whitespace, giữ `document_id`, tên nguồn, số trang và metadata chương. Hai cách chia chunk là A: cửa sổ 500 ký tự, overlap 100; B: theo đoạn, tối đa 500 ký tự. C truy vấn index B với filter `document_id` do câu hỏi cung cấp; C không tạo vector riêng. Embedding chạy local bằng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều). PostgreSQL 16 + pgvector xếp hạng theo cosine distance, với filter thực thi trong SQL.

## Kết quả đã lưu

| Cấu hình | Chunks | Hit@1 tổng | Gold-location Hit@3 tổng | Dev Hit@1 / Hit@3 | Test Hit@1 / Hit@3 | Coherence | Grounding test | Filter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — fixed | 287 | 6/20 (0.30) | 10/20 (0.50) | 4/15 / 6/15 | 2/5 / 4/5 | 17/20 | 6/10 | N/A |
| B — paragraph | 296 | 6/20 (0.30) | 10/20 (0.50) | 4/15 / 7/15 | 2/5 / 3/5 | 16/20 | 6/10 | N/A |
| C — B + filter | 296 (tái dùng B) | 6/20 (0.30) | 10/20 (0.50) | 4/15 / 7/15 | 2/5 / 3/5 | 16/20 | 6/10 | 4/4 |

Nguồn số liệu: `starter/results/metrics.json`, `chunk_coherence_audit.csv`, `grounding_audit.csv`, các trace A/B/C và `starter/artifacts/experiment_log.csv`. Thời gian indexing/evaluation và chi phí tài nguyên local không được đo; không suy đoán các số này. Không có chi phí API vì embedding chạy local.

## Đánh giá theo năm chiều

### 1. Độ chính xác retrieval

Hit@1 bằng 6/20 ở cả ba cấu hình; gold-location Hit@3 là 10/20 ở cả ba. B cải thiện dev Hit@3 từ 6/15 lên 7/15 nhưng test giảm từ 4/5 xuống 3/5. Điều này không chứng minh B kém tổng quát; mẫu test chỉ có năm câu và không được dùng để tinh chỉnh. C không cải thiện chỉ số tổng trên benchmark này.

Lỗi cụ thể: câu `q01` hỏi định nghĩa liên tục tại `ch1`, trang 20 nhưng top-3 trace A không có trang gold (lần lượt thấy các trang 36, 21, 31). Đây là lỗi tìm đúng vị trí, khác với trường hợp tìm đúng trang nhưng chunk chưa đủ bằng chứng.

### 2. Tính mạch lạc của chunk

Audit thủ công trên 10 vị trí cho A 17/20, B/C 16/20. Paragraph-aware không bảo đảm chunk tốt hơn khi PDF trích xuất làm mất cấu trúc. Audit ghi B bị cắt giữa câu về đạo hàm cấp 3 ở `ch3`, trang 32; công thức PDF khó đọc ở `ch1`, trang 56; ma trận Jacobi lỗi bố cục ở `ch4`, trang 53. Vì C dùng nguyên chunk B nên cùng điểm coherence.

### 3. Tính hữu dụng của metadata

Chunk giữ document ID, tên PDF, trang, chương, tiêu đề, tác giả và cơ sở đào tạo; ID chunk ổn định giúp truy trace. Filter C dùng `document_id` có nguồn từ câu hỏi trước truy xuất, và audit lưu compliance 4/4. SQL lọc trước khi xếp hạng; filter không dựa vào gold source/page. Hạn chế: metadata filter hiện chỉ phân biệt tài liệu, không tinh hơn tới section/loại nội dung; manifest chưa xác minh giấy phép và ngày truy cập.

### 4. Chất lượng grounding

Grounding là đánh giá thủ công xem top-3 có đủ điều kiện/ngữ cảnh để trả lời, không phải độ đúng của câu trả lời do LLM. Cả A/B/C đạt 6/10 trên năm câu test. Với `q14`, top-3 thiếu định nghĩa đầy đủ về khả vi và các điều kiện giới hạn. Với `q20`, top-3 thiếu định nghĩa tổng Riemann và điều kiện độc lập theo phân hoạch/điểm mẫu. `q05`, `q09`, `q18` đạt 2 điểm; `q14`, `q20` đạt 0 điểm cho mọi cấu hình. Chưa có LLM answer generation hoặc đánh giá câu trả lời.

### 5. Tác động chiến lược dữ liệu và đánh đổi

A dùng 287 chunks; B/C dùng 296, tăng 9 chunks. A tốt hơn coherence 1 điểm trên mẫu audit; B tốt hơn dev Hit@3 một câu nhưng giảm test Hit@3 một câu. C thể hiện filter tuân thủ 4/4 và tránh embedding lại B, nhưng không đổi kết quả tổng benchmark. Với ưu tiên truy trace và độ ổn định, A là baseline dễ giải thích; B là lựa chọn khi ranh giới đoạn quan trọng; C hữu ích khi ngữ cảnh cung cấp phạm vi tài liệu đáng tin cậy. Bằng chứng hiện tại không đủ để khẳng định chiến lược thắng tổng quát.

## Giới hạn, tính tái lập và quyền dữ liệu

- PDF trích xuất văn bản có thể làm hỏng ký hiệu, chỉ số, công thức nhiều dòng và bố cục ma trận. Ví dụ lỗi được ghi trong coherence/grounding audits. Có thể cần OCR hoặc xử lý công thức có kiểm chứng trước khi dùng corpus này cho câu hỏi phụ thuộc nặng vào ký hiệu.
- 20 câu và 5 câu test là mẫu nhỏ; số liệu chỉ áp dụng cho corpus, model và benchmark hiện tại.
- Các PDF chưa có thông tin quyền tái phân phối được xác minh. Manifest đánh dấu quyền là chưa rõ và cần xác minh; PDF, trace chứa nguyên văn chunk và input audit full-text được giữ local/ignore trong Git cho tới khi xác minh quyền.
- Thời gian chạy và chi phí điện/tài nguyên local: chưa đo. Không chạy lại indexing khi chỉ cần demo hoặc đánh giá.
- Công cụ AI hỗ trợ hoàn thiện code, giao diện và diễn giải artifacts trong phiên làm việc này. Kết quả số lấy từ files có sẵn; benchmark, trace và nhãn audit được giữ nguyên.

## Kết luận

Ba cấu hình có retrieval score tổng như nhau trong số liệu lưu. Paragraph chunking tăng nhẹ Hit@3 trên dev nhưng không cải thiện test; metadata filter tuân thủ đúng các yêu cầu đã audit và tái dùng B vectors. Nút thắt dễ thấy là độ đầy đủ/đọc được của evidence trích từ PDF và khả năng tìm đúng trang. Báo cáo không tuyên bố chất lượng trả lời LLM hoặc hiệu quả tổng quát ngoài corpus này.
