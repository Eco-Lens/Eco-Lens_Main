# Eco-Lens Semantic Mapping - ClimateBERT Scope Classifier

## 1. Tổng quan dự án

Module này là phần triển khai của bước Semantic Text Understanding trong pipeline Eco-Lens cho báo cáo ESG. Mục tiêu chính là phân loại các đoạn văn bản ESG đã được trích xuất thành 4 nhãn:

- Other
- Scope 1
- Scope 2
- Scope 3

Quá trình này được thực hiện bằng mô hình ClimateBERT fine-tuned cho bài toán sequence classification. Kết quả dự đoán sẽ được dùng cho các bước Semantic Mapping / Knowledge Base phía sau.

### Mục đích
- Phân loại văn bản ESG thành các nhóm phạm vi phát thải.
- Tạo dữ liệu đầu vào có nhãn cho các module sau.
- Cung cấp một mô hình có thể tái sử dụng cho inference.

### Đầu ra mong đợi
- Một mô hình classifier có thể dự đoán nhãn Scope cho đoạn văn bản đầu vào.
- Các file đánh giá như classification report, confusion matrix, predictions CSV.
- Các artifact huấn luyện được lưu sẵn để tiếp tục sử dụng.

### Vì sao dùng ClimateBERT
Notebook huấn luyện sử dụng mô hình ClimateBERT đã được pretrain trên dữ liệu liên quan đến khí hậu và môi trường, phù hợp hơn với văn bản ESG so với mô hình ngôn ngữ chung.

---

## 2. Vị trí trong Pipeline

Module này nằm ở giai đoạn phân loại ngữ nghĩa sau khi OCR và LayoutLMv3 đã tạo được các đoạn văn bản từ trang báo cáo.

```text
OCR
  ↓
LayoutLMv3
  ↓
ClimateBERT Scope Classification (Module hiện tại)
  ↓
Semantic Mapping
  ↓
Knowledge Base
  ↓
RAG
  ↓
LLM
  ↓
Phân tích ESG cuối cùng
```

---

## 3. Cấu trúc thư mục

```text
5.SemanticMapping/
├── Train_ClimateBERT_Scope.ipynb        # Notebook huấn luyện và đánh giá mô hình
├── Inference_ClimateBERT_Scope.ipynb    # Chạy inference Scope trên kết quả LayoutLMv3
├── Create_Scope_Dataset.ipynb           # Notebook tạo / chuẩn bị dataset
├── balance_scope_dataset.py             # Script làm sạch và cân bằng dataset
├── layoutlmv3_inference.py              # Script inference bằng LayoutLMv3 (không phải classifier Scope)
├── scope_dataset_balanced.csv           # Dataset huấn luyện đã được làm sạch và cân bằng
├── scope_dataset_balanced.xlsx          # Bản Excel tương ứng
├── label_scope_dataset.xlsx             # Dataset gốc có nhãn Scope
├── reason.md                            # Giải thích logic cân bằng dataset
├── ClimateBERT_Scope/                   # Thư mục checkpoint và mô hình tốt nhất
│   └── best_scope_classifier/          # Mô hình đã lưu sẵn cho inference
├── scope_dataset/                       # Bản sao dataset dùng cho huấn luyện
├── output/                              # Output từ các bước semantic mapping
├── semantic_output/                     # Output JSON/semantic blocks
├── faiss/                               # Index và metadata FAISS
├── ESG_KB.zip                           # Kho kiến thức ESG
├── ESG_Knowledge_Base.ipynb             # Notebook xây dựng / thao tác knowledge base
├── SemanticMapping_FAISS.ipynb          # Notebook cho FAISS / semantic mapping
└── 0_layoutlmv3_layout.json             # File layout từ LayoutLMv3
```

### Vai trò các thành phần quan trọng
- Notebook huấn luyện: thực hiện load dữ liệu, chia train/validation/test, tokenization, fine-tuning, đánh giá và lưu mô hình.
- Inference_ClimateBERT_Scope.ipynb: load mô hình ClimateBERT đã fine-tune, đọc file 0_layoutlmv3_layout.json, trích xuất các block text/figure, dự đoán Scope cho từng block, rồi sinh scope_predictions.json và scope_predictions.csv.
- Script balance_scope_dataset.py: làm sạch dataset, loại bỏ nội dung nhiễu, cân bằng phân bố lớp, tạo file CSV/XLSX đầu ra.
- Thư mục ClimateBERT_Scope: lưu checkpoint theo seed và mô hình tốt nhất.
- Thư mục output / semantic_output: chứa kết quả trung gian và semantic blocks dùng cho các bước tiếp theo.

---

## 4. Đầu vào

### Dataset đầu vào
Module hiện tại sử dụng dữ liệu được chuẩn bị từ các đoạn văn bản ESG. Trong notebook huấn luyện, dữ liệu được đọc từ:

- scope_dataset/scope_dataset_balanced.csv
- hoặc file CSV tương ứng trong thư mục gốc

### Các cột dùng cho training
Notebook huấn luyện chỉ sử dụng 2 cột chính:

- text
- scope

### Tiền xử lý dữ liệu
Trước khi huấn luyện, dữ liệu được:
- shuffle với random_state=42
- loại bỏ hàng có text rỗng hoặc chỉ toàn khoảng trắng
- giữ lại các cột cần thiết
- gán nhãn số bằng mapping:
  - Other -> 0
  - Scope 1 -> 1
  - Scope 2 -> 2
  - Scope 3 -> 3

### Chia Train / Validation / Test
Notebook dùng phương pháp train_test_split theo stratify trên label:
- Train: 80%
- Validation: 10%
- Test: 10%

Cụ thể:
1. Chia ban đầu train 80% và temp 20%
2. Chia temp thành validation 50% và test 50%

### Label Mapping

| Label gốc | Label số |
|---|---:|
| Other | 0 |
| Scope 1 | 1 |
| Scope 2 | 2 |
| Scope 3 | 3 |

### Dataset gốc cho việc cân bằng
Script balance_scope_dataset.py đọc file Excel gốc label_scope_dataset.xlsx và yêu cầu các cột như:
- block_id
- page
- bbox
- type
- confidence
- semantic_standard
- semantic_document
- semantic_score
- matched_chunk
- matched_page
- matched_document
- text
- scope

---

## 5. Kiến trúc mô hình

### Base model
- climatebert/distilroberta-base-climate-f

### Tokenizer
- AutoTokenizer từ cùng model

### Mục tiêu bài toán
- Sequence Classification
- Số lớp đầu ra: 4 lớp

### Cách hoạt động
- Văn bản đầu vào được tokenize và padding/truncation về max_length 256.
- Mô hình dự đoán xác suất cho 4 nhãn Scope.
- Label được chuyển thành id và lưu trong label_mapping.json.

### Vì sao phù hợp
Mô hình này phù hợp vì:
- được pretrain trên dữ liệu liên quan đến khí hậu và môi trường
- có khả năng xử lý văn bản ngắn/đến trung bình như câu hoặc đoạn ESG
- dễ tích hợp vào pipeline Hugging Face hiện có

---

## 6. Quy trình huấn luyện

Pipeline huấn luyện được triển khai như sau:

```text
Dataset
  ↓
Preprocessing
  ↓
Tokenization
  ↓
ClimateBERT Fine-tuning
  ↓
Validation
  ↓
Testing
  ↓
Best Model Selection
  ↓
Save Artifacts
```

### Các bước thực tế trong notebook
1. Tải dataset từ file CSV.
2. Làm sạch và loại bỏ dòng text rỗng.
3. Gán nhãn số bằng label mapping.
4. Chia train/validation/test bằng stratified split.
5. Chuyển dataset sang Hugging Face Dataset.
6. Tokenize bằng tokenizer.
7. Huấn luyện 5 seed khác nhau: 42, 123, 2026, 3407, 8888.
8. Đánh giá trên validation và test.
9. Chọn best model theo macro F1 trên validation.
10. Lưu mô hình tốt nhất và các file báo cáo.

Sau khi huấn luyện xong, luồng suy luận sẽ tiếp tục như sau:

```text
Best Scope Classifier
  ↓
Inference_ClimateBERT_Scope.ipynb
  ↓
0_layoutlmv3_layout.json
  ↓
Extract text blocks
  ↓
Scope Prediction
  ↓
scope_predictions.json
scope_predictions.csv
```
---

## 7. Cấu hình huấn luyện

Các hyperparameter quan trọng được tìm thấy trong notebook:

| Tham số | Giá trị |
|---|---:|
| Base model | climatebert/distilroberta-base-climate-f |
| Max length | 256 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Epochs | 20 |
| Optimizer | AdamW (adamw_torch) |
| Evaluation strategy | epoch |
| Save strategy | epoch |
| Logging strategy | epoch |
| Early stopping patience | 2 |
| Multi-seed training | Có (5 seeds) |
| Resume training | Có |
| Mixed precision | fp16 nếu CUDA có sẵn |

### Các tùy chọn huấn luyện khác
- load_best_model_at_end=True
- metric_for_best_model="macro_f1"
- greater_is_better=True
- save_total_limit=1
- report_to="wandb"

---

## 8. Các file đầu ra

Các file đầu ra chính được sinh ra trong thư mục best_scope_classifier:

| File | Chức năng |
|---|---|
| config.json | Cấu hình mô hình Hugging Face |
| model.safetensors | Trọng số mô hình |
| tokenizer.json | Tokenizer JSON |
| tokenizer_config.json | Cấu hình tokenizer |
| training_config.json | Siêu tham số và thông tin huấn luyện |
| label_mapping.json | Bản đồ giữa nhãn và id |
| training_summary.csv | Kết quả huấn luyện theo từng seed |
| results_summary.csv | Tổng hợp thống kê trung bình / độ lệch chuẩn |
| predictions.csv | Dự đoán trên tập test với true label, pred label và confidence |
| classification_report.txt | Báo cáo phân loại theo text |
| classification_report.json | Báo cáo phân loại theo JSON |
| confusion_matrix.png | Biểu đồ confusion matrix |
| confusion_matrix.csv | Ma trận nhầm lẫn dạng CSV |
| scope_predictions.csv | Kết quả dự đoán Scope cho toàn bộ block từ đầu ra LayoutLMv3 |
| scope_predictions.json | Kết quả inference dạng JSON phục vụ pipeline tiếp theo |
| README.txt | Tóm tắt nhanh về mô hình và môi trường |

### Thư mục checkpoint
Bên cạnh mô hình tốt nhất, notebook còn lưu checkpoint theo từng seed trong thư mục:
- ClimateBERT_Scope/seed_42
- ClimateBERT_Scope/seed_123
- ClimateBERT_Scope/seed_2026
- ClimateBERT_Scope/seed_3407
- ClimateBERT_Scope/seed_8888

---

## 9. Tích hợp WandB

Notebook có tích hợp Weights & Biases để theo dõi và lưu kết quả huấn luyện.

### Những thứ được ghi lên WandB
- Training loss
- Validation loss
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision
- Weighted Recall
- Weighted F1
- Confusion Matrix
- Classification Report
- Model artifact

### Artifact được upload
- best_scope_classifier (dạng model artifact)
- confusion_matrix.png
- training_summary.csv
- results_summary.csv
- predictions.csv
- confusion_matrix.csv
- classification_report.json
- README.txt

---

## 10. Resume Training

Notebook có cơ chế tiếp tục huấn luyện từ checkpoint khi quá trình chạy bị gián đoạn.

### Cách hoạt động
- Mỗi seed có thư mục output riêng: seed_{seed}
- Nếu thư mục checkpoint tồn tại, trainer sẽ gọi get_last_checkpoint để tìm checkpoint mới nhất.
- Nếu file completed.txt tồn tại, notebook sẽ bỏ qua seed đó.
- Nếu không có checkpoint, huấn luyện bắt đầu từ đầu.

### Ứng dụng thực tế
Cơ chế này phù hợp khi:
- phiên Colab bị ngắt
- quá trình huấn luyện bị treo
- cần tiếp tục chạy từ checkpoint gần nhất thay vì bắt đầu lại

> Chưa được triển khai (Not implemented): chưa thấy script CLI riêng để resume từ dòng lệnh ngoài notebook.

---

## 11. Inference Pipeline

Sau khi huấn luyện, mô hình có thể được dùng để suy luận trực tiếp trên đầu ra của LayoutLMv3.

### Input
File đầu vào cho luồng inference là:

- 0_layoutlmv3_layout.json

File này được tạo ở bước LayoutLMv3 và chứa các block đã được nhận dạng từ trang báo cáo.

### Load Model
Notebook Inference_ClimateBERT_Scope.ipynb sẽ load:

- tokenizer
- ClimateBERT classifier đã fine-tune
- label_mapping.json

### Extract Text Blocks
Notebook chỉ xử lý các block có kiểu:

- text
- figure

Các block loại:

- table
- header
- footer
- ignore
- các loại khác

sẽ được bỏ qua trong quá trình dự đoán Scope.

### Scope Prediction
Mỗi block được dự đoán thành:

- Scope
- Scope ID
- Confidence
- Probability của từng lớp

### Output
Notebook sinh ra các file kết quả:

- scope_predictions.csv
- scope_predictions.json

### Ví dụ output
```json
{
  "page": 15,
  "block_id": 8,
  "label": "text",
  "text": "Purchased electricity...",
  "scope": "Scope 2",
  "scope_id": 2,
  "confidence": 0.987,
  "probabilities": {
    "Other": 0.001,
    "Scope 1": 0.004,
    "Scope 2": 0.987,
    "Scope 3": 0.008
  }
}
```

---

## 12. Tái sử dụng mô hình đã huấn luyện

Mô hình đã huấn luyện có thể được tái sử dụng bằng cách load lại tokenizer và model từ thư mục best_scope_classifier.

Notebook Inference_ClimateBERT_Scope.ipynb cung cấp ví dụ hoàn chỉnh cho việc:
- đọc đầu ra LayoutLMv3
- tokenize tự động
- batch inference
- lưu CSV
- lưu JSON
- tính confidence

### Cách dùng cơ bản
- Dùng AutoTokenizer.from_pretrained(path)
- Dùng AutoModelForSequenceClassification.from_pretrained(path)
- Tokenize đoạn văn bản đầu vào
- Chạy inference và lấy nhãn có xác suất cao nhất

### Ví dụ ngắn
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "ClimateBERT_Scope/best_scope_classifier"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

---

## 13. Tích hợp với các module khác

### Input
Luồng dữ liệu đầu vào cho module này như sau:

```text
0_layoutlmv3_layout.json
  ↓
Inference_ClimateBERT_Scope.ipynb
  ↓
scope_predictions.json
  ↓
Semantic Mapping
  ↓
Knowledge Base Retrieval
  ↓
FAISS
  ↓
RAG
  ↓
LLM ESG Analysis
```

### Output
Kết quả dự đoán Scope sẽ phục vụ cho các module phía sau:
- Semantic Mapping
- Knowledge Base Retrieval
- FAISS / retrieval
- RAG / LLM cho phân tích ESG cuối cùng

---

## 14. Xử lý sự cố

Các lỗi thường gặp và cách xử lý:

- GPU Out Of Memory (OOM)
  - giảm batch size
  - giảm max_length
  - dùng CPU nếu cần

- Khôi phục từ checkpoint
  - dùng thư mục seed_x và resume_from_checkpoint

- Đăng nhập WandB
  - chạy wandb.login() trước khi huấn luyện

- Không tương thích phiên bản Transformers
  - notebook có kiểm tra version và dùng TrainingArguments phù hợp

- Sai định dạng dataset
  - đảm bảo cột text và scope tồn tại
  - đảm bảo text không rỗng

- Thiếu file model hoặc tokenizer
  - kiểm tra thư mục best_scope_classifier và checkpoint tương ứng

- Colab / Google Drive path mismatch
  - notebook hiện đang dùng đường dẫn /content/drive/MyDrive/...

---

## 15. Thư viện phụ thuộc

Các thư viện Python quan trọng được sử dụng:

- pandas
- numpy
- torch
- datasets
- transformers
- scikit-learn
- matplotlib
- wandb
- accelerate
- evaluate
- sentencepiece

---

## 16. Hướng cải tiến trong tương lai

Dựa trên mã nguồn hiện tại, các hướng cải tiến có thể gồm:

- ✅ Đã có notebook inference cho batch prediction từ LayoutLMv3.
- ⏳ Chưa có CLI hoặc REST API để chạy production.
- ⏳ Chưa tối ưu batch inference cho GPU trên nhiều tài liệu.
- ⏳ Có thể bổ sung ngưỡng confidence để đánh dấu các dự đoán cần xem xét thủ công.
- ⏳ Có thể chuẩn hóa thêm schema đầu vào/đầu ra cho pipeline tiếp theo.
- ⏳ Có thể so sánh các base model khác ngoài ClimateBERT.

---

## Kết luận

Module này là một phần quan trọng của pipeline Eco-Lens để biến các đoạn văn bản ESG thành nhãn Scope 1/2/3/Other. Với mô hình ClimateBERT fine-tuned và luồng inference trên đầu ra LayoutLMv3, module này cung cấp một cơ chế phân loại ngữ nghĩa có thể tái sử dụng cho các bước xử lý, truy xuất và phân tích ESG phía sau.
