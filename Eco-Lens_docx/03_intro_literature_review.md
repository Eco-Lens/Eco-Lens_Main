# Introduction and Literature Review

## 1. Introduction and Background

### 1.1 Objective

Dự án nhằm xây dựng một hệ thống Trí tuệ nhân tạo có khả năng tự động hóa toàn diện quá trình phân tích báo cáo ESG (Môi trường, Xã hội và Quản trị), với trọng tâm hiện tại là phát triển **Module 1: Multimodal ESG Report Understanding**. Hệ thống có khả năng tiếp nhận báo cáo dưới dạng PDF hoặc tài liệu số hóa, sử dụng công nghệ OCR kết hợp mô hình LayoutLMv3 để xử lý cấu trúc tài liệu đa phương thức, từ đó trích xuất chính xác dữ liệu phát thải Carbon và phân loại theo Scope 1, Scope 2 và Scope 3 dựa trên tiêu chuẩn GHG Protocol, đồng thời đối chiếu với các bộ tiêu chuẩn quốc tế như GRI Standards thông qua công nghệ RAG.

Mục tiêu cốt lõi của dự án là tạo ra một hệ thống tự động hóa quá trình phân tích báo cáo, gia tăng tính minh bạch của dữ liệu Carbon và hỗ trợ ra quyết định chiến lược cho ban lãnh đạo doanh nghiệp, nhà đầu tư, kiểm toán viên và các cơ quan quản lý.

### 1.2 Motivation

Trong bối cảnh biến đổi khí hậu ngày càng trở nên nghiêm trọng, các yêu cầu về minh bạch ESG cùng các quy định liên quan đến phát thải Carbon đang buộc các doanh nghiệp phải công bố dữ liệu môi trường một cách chính xác, có thể kiểm chứng và dễ dàng giải thích. Báo cáo ESG hiện nay đã vượt ra khỏi khuôn khổ truyền thông thương hiệu đơn thuần, tác động trực tiếp đến:

- **Tuân thủ pháp luật (compliance):** Nhiều quốc gia, đặc biệt là tại châu Âu, đã đưa ra quy định bắt buộc về công bố thông tin ESG đối với doanh nghiệp đạt đến quy mô nhất định.
- **Đánh giá từ nhà đầu tư:** Nhà đầu tư hiện đại xem xét rủi ro ESG song song với doanh thu và lợi nhuận. Dữ liệu ESG ảnh hưởng trực tiếp đến đánh giá rủi ro tài chính dài hạn, bao gồm rủi ro thuế Carbon, mức độ minh bạch của hệ thống quản trị và rủi ro tiềm ẩn trong chuỗi cung ứng.
- **Kiểm toán bền vững:** Các kiểm toán viên cần công cụ có khả năng truy vết dữ liệu để xác minh nguồn gốc dự đoán của AI, đảm bảo tính xác thực của báo cáo ESG.
- **Khả năng tham gia chuỗi cung ứng quốc tế:** Nhiều đối tác quốc tế yêu cầu minh bạch ESG, doanh nghiệp không đáp ứng được có thể mất hợp đồng và mất khả năng tham gia chuỗi cung ứng toàn cầu.

Mặc dù đóng vai trò quan trọng, các báo cáo ESG thường có cấu trúc vô cùng phức tạp với sự đan xen của văn bản, bảng biểu số liệu, biểu đồ xu hướng và nhiều thuật ngữ chuyên ngành. Việc tiếp cận, đọc hiểu, trích xuất và đánh giá thủ công những tài liệu này không chỉ tiêu tốn nguồn lực khổng lồ mà còn khó đạt được sự chuẩn hóa, đồng thời tiềm ẩn nguy cơ sai lệch thông tin hoặc các hành vi "tẩy xanh" (greenwashing) — nơi doanh nghiệp sử dụng ngôn từ truyền thông tích cực như "eco-friendly" hay "sustainable" nhưng dữ liệu định lượng thực tế lại cho thấy điều ngược lại.

Dự án này ra đời nhằm giải quyết bốn rào cản kỹ thuật và chuyên môn lớn: (1) sự phân tán và khó chuẩn hóa của dữ liệu ESG, (2) nguy cơ nhầm lẫn trong phân loại phát thải Carbon theo Scope, (3) rào cản về niềm tin đối với các mô hình AI "hộp đen" thiếu khả năng giải thích, và (4) sự tinh vi của các hành vi greenwashing ngày càng khó phát hiện.

### 1.3 Background Information

Hệ thống được thiết kế dưới dạng một đường ống dữ liệu (pipeline) toàn diện, chuyển hóa dữ liệu ESG thô từ các báo cáo phức tạp thành những phân tích và dự báo mang tính chiến lược. Luồng xử lý của Module 1 được chia thành năm bước:

1. **OCR (Text Extraction):** Sử dụng PaddleOCR để chuyển đổi PDF/ảnh quét thành văn bản máy đọc được, với khả năng đọc bảng biểu mạnh mẽ và hoạt động ổn định với tài liệu quét phức tạp.
2. **Layout Understanding:** Áp dụng LayoutLMv3 — mô hình đa phương thức tiên tiến cho phép AI kết hợp đồng thời hình ảnh tài liệu, nội dung văn bản và vị trí không gian để phân biệt tiêu đề, đoạn văn, biểu đồ và các khối dữ liệu ESG quan trọng.
3. **Table Understanding:** Sử dụng Microsoft Table Transformer (DETR-based) để phân tích cấu trúc bảng, xử lý các ô gộp (merge cell) và tiêu đề nhiều dòng, đảm bảo số liệu định lượng được gắn chính xác với hạng mục tương ứng.
4. **Semantic Mapping (Scope Classification):** Tích hợp ClimateBERT — mô hình được huấn luyện chuyên biệt trên bộ dữ liệu về biến đổi khí hậu — để tự động đọc hiểu ngữ cảnh và phân loại phát thải vào Scope 1, Scope 2 hay Scope 3.
5. **RAG Standards Retrieval:** Sử dụng BGE-large kết hợp RAG để truy xuất hướng dẫn từ GHG Protocol và GRI Standards, xác thực chéo các quyết định phân loại và chuẩn hóa dữ liệu đầu ra.

Về chiến lược dữ liệu, dự án sử dụng tập dữ liệu nội địa gồm hơn 300 báo cáo phát triển bền vững từ các doanh nghiệp Việt Nam, bám sát bối cảnh pháp lý địa phương như Nghị định 06/2022/NĐ-CP về giảm phát thải khí nhà kính. Để khắc phục hạn chế về quy mô, dự án áp dụng chiến lược kết hợp đa nguồn (Multi-source Augmentation) với các tập dữ liệu NLP chuyên ngành ESG từ Hugging Face, kết nối vĩnh viễn với kho tri thức GHG Protocol và GRI Standards qua RAG, và tích hợp thêm bộ dữ liệu pháp lý quốc tế như Climate Change Laws of the World.

Dự án được triển khai trong vòng 16 tuần với bốn giai đoạn: (1) thu thập và tiền xử lý dữ liệu với PaddleOCR và LayoutLMv3, (2) tinh chỉnh Table Transformer cho trích xuất bảng biểu phát thải, (3) tích hợp ClimateBERT cho phân loại Scope và RAG cho đối chiếu tiêu chuẩn, (4) lắp ráp pipeline hoàn chỉnh và kiểm thử.

---

## 2. Literature Review

### 2.1 Document Understanding và OCR cho Tài liệu ESG

Việc trích xuất thông tin từ tài liệu có cấu trúc phức tạp đã là một chủ đề nghiên cứu lâu dài trong lĩnh vực Document AI. Các hệ thống OCR truyền thống như Tesseract OCR (Smith, 2007) có khả năng nhận dạng ký tự trên văn bản thuần, nhưng thường thất bại trước các tài liệu có cấu trúc hỗn hợp như báo cáo ESG với văn bản đa cột, bảng biểu lồng ghép và biểu đồ xen kẽ (Zhong và cộng sự, 2019).

PaddleOCR (Du và cộng sự, 2020) nổi lên như một giải pháp vượt trội với kiến trúc Detection (DBNet) kết hợp Recognition (Transformer), đặc biệt mạnh trong việc nhận dạng cấu trúc bảng biểu. So với các lựa chọn khác như EasyOCR, TrOCR (Li và cộng sự, 2023) hay các dịch vụ đám mây như AWS Textract hay Google Document AI, PaddleOCR có lợi thế về khả năng đọc bảng, mã nguồn mở, độ chính xác cao, hỗ trợ đa ngôn ngữ và dễ tích hợp pipeline, dù yêu cầu thiết lập phức tạp hơn Tesseract.

Các thách thức OCR điển hình với báo cáo ESG bao gồm: scan PDF chất lượng thấp (mờ, font nhỏ), bố cục đa cột gây đọc sai thứ tự, bảng lồng nhau, nội dung hỗn hợp giữa text và hình ảnh, sự khác biệt về layout giữa các báo cáo gây khó khăn cho việc tổng quát hóa (generalization).

### 2.2 Mô hình Đa phương thức trong Document AI

LayoutLMv3 (Huang và cộng sự, 2022) đại diện cho bước tiến quan trọng trong Document AI khi kết hợp đồng thời ba thành phần: text embedding (nội dung văn bản), position embedding (vị trí trên trang) và image embedding (hình ảnh tài liệu). Khác với các phiên bản trước như LayoutLMv2 (Xu và cộng sự, 2021) vốn có khả năng visual yếu hơn, LayoutLMv3 sử dụng kiến trúc multimodal transformer với khả năng hiểu ngữ nghĩa bố cục vượt trội, đặc biệt phù hợp với tài liệu ESG có cấu trúc layout phức tạp.

So sánh với các mô hình khác: DocFormer (Appalaraju và cộng sự, 2021) có khả năng hiểu tài liệu ngữ nghĩa tốt nhưng cộng đồng phát triển nhỏ; Donut (Kim và cộng sự, 2022) là mô hình end-to-end không cần OCR nhưng việc tinh chỉnh (fine-tune) khó và không ổn định; DiT (Li và cộng sự, 2022) có khả năng visual mạnh nhưng không mạnh về ngữ nghĩa văn bản; Detectron2 kết hợp PubLayNet (Zhong và cộng sự, 2019) chỉ phát hiện layout object mà không hiểu ngữ nghĩa. LayoutLMv3 được đánh giá có độ phù hợp rất cao cho bài toán ESG document understanding nhờ khả năng multimodal, table-aware và semantic layout vượt trội.

### 2.3 Table Understanding cho Dữ liệu ESG

Trích xuất dữ liệu từ bảng biểu ESG đặt ra nhiều thách thức đặc thù: ô gộp (merge cell) gây sai mapping, tiêu đề nhiều dòng (multi-row header) làm sai cột, bảng lồng nhau (nested table) gây lỗi parse, OCR lệch dòng dẫn đến phân loại Scope sai, và đơn vị đo khác cột gây mapping sai.

Microsoft Table Transformer (Smock và cộng sự, 2022), dựa trên kiến trúc DETR (Carion và cộng sự, 2020), là mô hình được đánh giá cao nhất cho bài toán này. So với các giải pháp khác: CascadeTabNet (Prasad và cộng sự, 2020) chỉ phát hiện table mà không hiểu semantic; Camelot (rule-based) và Tabula (rule-based) có độ chính xác thấp với scan PDF; DeepDeSRT (Schreiber và cộng sự, 2017) parsing tốt nhưng khó huấn luyện. Table Transformer vượt trội nhờ khả năng hiểu row/column relation, xử lý merge cell, phân tích cấu trúc bảng ngữ nghĩa, và đặc biệt phù hợp với bảng ESG phức tạp.

### 2.4 NLP cho Phân loại Phát thải và ESG

ClimateBERT (Webersinke và cộng sự, 2022) là mô hình BERT được tiền huấn luyện trên các bộ dữ liệu chuyên biệt về biến đổi khí hậu, giúp nó hiểu được các thuật ngữ như carbon emission, sustainability, renewable energy, supply chain emission, net zero. Khả năng semantic classification của ClimateBERT cho phép nó nhận biết rằng "Indirect emissions from purchased electricity" thực chất thuộc Scope 2 — một nhiệm vụ mà các mô hình NLP thông thường không làm được nếu thiếu kiến thức domain về GHG Protocol.

So sánh với các mô hình khác: BERT (Devlin và cộng sự, 2019) và RoBERTa (Liu và cộng sự, 2019) có năng lực NLP mạnh nhưng thiếu kiến thức ESG-specific; DeBERTa-v3 (He và cộng sự, 2021) có context understanding cực mạnh nhưng yêu cầu GPU lớn; FinBERT (Huang và cộng sự, 2020) hiểu tốt lĩnh vực tài chính nhưng không đủ mạnh cho ESG; SciBERT (Beltagy và cộng sự, 2019) phù hợp với văn bản kỹ thuật nhưng không climate-focused. SVM kết hợp TF-IDF có ưu điểm dễ huấn luyện nhưng hoàn toàn không hiểu ngữ nghĩa. ClimateBERT được đánh giá có độ phù hợp rất cao nhờ khả năng hiểu ngữ nghĩa ESG và thuật ngữ khí hậu vượt trội.

### 2.5 Retrieval-Augmented Generation (RAG) trong ESG

RAG (Lewis và cộng sự, 2020) là kiến trúc kết hợp retriever và generator, cho phép hệ thống truy xuất thông tin từ kho tri thức bên ngoài trước khi đưa ra dự đoán. Trong bối cảnh ESG, RAG cho phép đối chiếu dữ liệu trích xuất với các tiêu chuẩn quốc tế như GHG Protocol và GRI Standards, giúp chuẩn hóa thông tin và giảm nguy cơ phân loại sai Scope phát thải.

BGE-large (Xiao và cộng sự, 2023) là mô hình embedding mạnh mẽ cho dense retrieval, được chọn nhờ khả năng semantic retrieval vượt trội, xử lý tài liệu dài tốt, chất lượng embedding cao và phù hợp với retrieval ESG standard. So với các lựa chọn khác: BM25 chỉ dựa trên keyword, không hiểu ngữ nghĩa; Sentence-BERT (Reimers và Gurevych, 2019) có embedding chất lượng thấp hơn; DPR (Karpukhin và cộng sự, 2020) có fine-tune phức tạp; các API embedding như OpenAI phụ thuộc vào dịch vụ bên ngoài.

### 2.6 Khoảng trống Nghiên cứu (Research Gaps)

Mặc dù các công nghệ thành phần đã có những bước tiến đáng kể, vẫn tồn tại những khoảng trống quan trọng:

1. **Thiếu hệ thống tích hợp end-to-end cho ESG:** Hầu hết các nghiên cứu hiện tại tập trung vào từng bước riêng lẻ (OCR, layout understanding, table parsing, NLP classification) mà chưa có một pipeline hoàn chỉnh kết hợp tất cả thành phần để xử lý báo cáo ESG một cách tự động.

2. **Hạn chế về dữ liệu ESG ngữ cảnh địa phương:** Các tập dữ liệu ESG quốc tế lớn (SEC EDGAR, Kaggle) không bám sát môi trường pháp lý và đặc thù vận hành của doanh nghiệp tại thị trường Việt Nam (ví dụ: Nghị định 06/2022/NĐ-CP). Việc áp dụng trực tiếp các mô hình huấn luyện trên dữ liệu quốc tế vào bối cảnh Việt Nam có thể dẫn đến sai lệch đáng kể.

3. **Thiếu cơ chế truy vết trong phân tích ESG:** Các hệ thống hiện có thường dựa trên mô hình "hộp đen", đưa ra kết quả phân loại mà không có khả năng truy vết nguồn gốc dữ liệu. Điều này khiến doanh nghiệp, kiểm toán viên và nhà đầu tư khó tin tưởng vào kết quả.

### 2.7 Tổng hợp Mô hình

| Mốc xử lý | Mô hình được chọn | Vai trò |
|-----------|-------------------|---------|
| OCR | PaddleOCR | Trích xuất văn bản từ ảnh/PDF |
| Layout Understanding | LayoutLMv3 | Hiểu bố cục tài liệu ESG |
| Table Understanding | Microsoft Table Transformer | Parse bảng biểu ESG |
| Semantic Mapping (Scope) | ClimateBERT | Phân loại Scope 1/2/3 |
| RAG Retrieval | BGE-large | Truy xuất chuẩn GHG/GRI |
