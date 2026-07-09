# Báo cáo quá trình huấn luyện và đánh giá LayoutLMv3 sau bước merge OCR với region label

**Phạm vi báo cáo:** Tài liệu này ghi lại toàn bộ quy trình từ sau khi đã merge OCR words/bboxes với label region annotation cho đến khi fine-tune và đánh giá mô hình `LayoutLMv3ForTokenClassification`.

**Bài toán:** Token classification / word-level layout region classification cho báo cáo ESG / Annual Report.

---

## 1. Trạng thái đầu vào sau bước merge OCR với label region

Sau khi hoàn thành các bước:

```text
PDF / ESG Report
→ render thành ảnh page
→ annotate layout region trên Roboflow
→ export COCO annotation
→ PaddleOCR lấy OCR words + bbox
→ merge OCR word với COCO region bằng center point
→ gán label layout cho từng OCR word
→ normalize bbox về thang 0–1000
```

ta thu được file dataset chính dùng để train LayoutLMv3:

```text
0_layoutlmv3_dataset_normalized.json
```

Trong Colab, file này được đặt tại:

```python
DATASET_DIR = "/content/layoutlmv3_data/valid/"
IMAGE_DIR = "/content/layoutlmv3_data/valid/images"
JSON_PATH = "/content/layoutlmv3_data/valid/0_layoutlmv3_dataset_normalized.json"
```

Mỗi phần tử trong JSON tương ứng với một page sample:

```json
{
  "image_id": 0,
  "file_name": "bao_viet_holdings_2024_p345_jpg.rf.a30d88b69fd1694f4aa1e448d0835a10.jpg",
  "width": 2382,
  "height": 3367,
  "words": ["Bao Viet Holdings", "B09-DN/HN", "..."],
  "bboxes": [[131, 42, 322, 63], [787, 46, 883, 63]],
  "labels": ["header", "ignore"],
  "bbox_scale": "0-1000"
}
```

Ý nghĩa các trường:

| Trường | Ý nghĩa |
|---|---|
| `image_id` | ID của page/image |
| `file_name` | Tên ảnh sạch tương ứng |
| `width`, `height` | Kích thước ảnh gốc, dùng để kiểm tra alignment |
| `words` | Danh sách OCR word/line được trích xuất từ PaddleOCR |
| `bboxes` | Bbox của từng OCR word/line, đã normalize về thang 0–1000 |
| `labels` | Label layout tương ứng với từng OCR word/line |
| `bbox_scale` | Xác nhận bbox đang ở scale 0–1000 |

Điều kiện bắt buộc:

```text
len(words) == len(bboxes) == len(labels)
```

---

## 2. Quy tắc gán label sau khi merge OCR với region annotation

Quy tắc merge được sử dụng:

```text
1. Lấy bbox của từng OCR word/line từ PaddleOCR.
2. Tính center point của bbox OCR.
3. Kiểm tra center point nằm trong region annotation nào của COCO.
4. Nếu center point nằm trong một region, gán label của region đó cho OCR word.
5. Nếu center point nằm trong nhiều region, ưu tiên region có diện tích nhỏ hơn.
6. Nếu center point không nằm trong region nào, gán label O.
7. Normalize bbox từ pixel gốc sang thang 0–1000.
```

Lý do dùng center point:

```text
Center point giúp quyết định OCR word thuộc vùng layout nào một cách ổn định hơn so với chỉ kiểm tra overlap một phần bbox.
```

Lý do ưu tiên region nhỏ hơn:

```text
Khi vùng annotation bị lồng nhau hoặc giao nhau, region nhỏ hơn thường mang ý nghĩa cụ thể hơn.
Ví dụ: table_text nằm trong table hoặc text box nằm trong một vùng lớn hơn.
```

---

## 3. Các label dùng để train LayoutLMv3

Tổng cộng có 10 label. Trong đó `O` là label nền cho token không nằm trong bất kỳ region annotation nào, còn 9 label còn lại là các vùng layout được annotate theo rule cụ thể.

| ID | Label | Tier | Vai trò trong pipeline |
|---:|---|---|---|
| 0 | `O` | Background | Token không nằm trong bất kỳ region annotation nào |
| 1 | `chart` | Tier 1 | Vùng biểu đồ, dùng cho Chart Understanding |
| 2 | `figure` | Tier 2 | Ảnh có chữ/có ý nghĩa hoặc bảng bị lẫn icon/hình ảnh không phù hợp |
| 3 | `footer` | None Tier | Vùng cuối trang, không dùng cho downstream extraction |
| 4 | `header` | Tier 3 | Các title, chữ in hoa đậm, tiêu đề chính |
| 5 | `ignore` | None Tier | Vùng nhiễu, logo, icon, hình nền, thông tin không cần xử lý |
| 6 | `table` | Tier 1 | Bảng hoàn chỉnh có số liệu, ưu tiên cho Table Understanding |
| 7 | `table_text` | Tier 3 | Bảng dạng chữ, ít/không có số liệu quan trọng |
| 8 | `text` | Tier 2 | Cụm text chính, gồm tên mục và nội dung mục |
| 9 | `toc` | Tier 4 | Table of Content / mục lục / danh sách nội dung |

### 3.1. Rule chi tiết cho từng label

#### 1. `chart` — Tier 1

`chart` dùng cho các vùng biểu đồ. Khi annotate, cần phân tách từng biểu đồ nhỏ nếu trên trang có nhiều biểu đồ. Không nên đóng khung nhiều hơn 2 biểu đồ vào cùng một label, vì điều này làm giảm độ chính xác khi đưa sang bước Chart Understanding.

Vai trò sau LayoutLMv3:

```text
chart → crop vùng biểu đồ → Chart Understanding → trích xuất xu hướng, trục, legend, giá trị hoặc thông tin định lượng phụ trợ
```

#### 2. `figure` — Tier 2

`figure` dùng cho ảnh có chữ và có ý nghĩa. Nếu một vùng nhìn giống table nhưng bị lẫn icon, hình ảnh phụ, hình minh họa hoặc thành phần tạp khiến nó không còn là bảng số liệu sạch, vùng đó được đánh thành `figure` thay vì `table`.

Ví dụ nên đánh `figure`:

```text
- Infographic có chữ
- Ảnh minh họa có text quan trọng
- Bảng bị lẫn icon/hình ảnh tạp
- Vùng visual có chữ nhưng không phải chart/table chuẩn
```

#### 3. `footer` — None Tier

`footer` không dùng cho downstream extraction. Label này thường nằm ở phía dưới trang và có tính chất tương tự `ignore`. Nội dung footer có thể gồm thông tin công ty, chữ ký lãnh đạo, thông tin về người lãnh đạo hoặc các thông tin phụ ở cuối trang.

Vai trò chính là giúp model học để loại bỏ các vùng cuối trang ít liên quan đến Carbon extraction.

#### 4. `header` — Tier 3

`header` dùng cho các title, phần chữ in hoa đậm và các tiêu đề chính. Trong guideline hiện tại, `header` không chỉ mang nghĩa vùng đầu trang theo vị trí, mà còn mang nghĩa các vùng tiêu đề nổi bật trong cấu trúc tài liệu.

Ví dụ nên đánh `header`:

```text
- Tiêu đề chương/mục chính
- Chữ IN HOA ĐẬM
- Title nổi bật trên trang
- Heading dùng để chia cấu trúc báo cáo
```

#### 5. `ignore` — None Tier

`ignore` là vùng không dùng cho downstream extraction. Label này dùng cho các thành phần nhiễu, không có giá trị phân tích ESG/Carbon hoặc có thể làm sai pipeline nếu đưa vào xử lý.

Các trường hợp đánh `ignore`:

```text
- Vùng có chứa từ tiếng Việt nhưng không phục vụ mục tiêu extraction
- Tên công ty khi chỉ đóng vai trò branding
- Logo công ty
- Số trang
- Hình ảnh không liên quan
- Mọi hình ảnh không có chữ và không có ý nghĩa, chỉ làm nền cho báo cáo đẹp hơn
- Icon dư thừa
- Phần thừa khác biệt không cần xử lý
- Logo công ty hợp tác hoặc đối tác khác
- Cúp, biểu tượng giải thưởng, icon trang trí
- Header được cách điệu bằng ảnh nhưng không cần dùng cho phân tích
```

#### 6. `table` — Tier 1

`table` dùng cho bảng hoàn chỉnh có số liệu. Đây là một trong các label quan trọng nhất của pipeline vì dữ liệu bảng thường là nguồn chính xác nhất để trích xuất Carbon footprint.

Rule đánh `table`:

```text
- Là một bảng hoàn chỉnh
- Có số liệu
- Có thể lấy cả tiêu đề bảng
- Có thể lấy cả đơn vị đo lường nếu nằm trong vùng bảng
```

Vai trò sau LayoutLMv3:

```text
table → crop/gom vùng bảng → Table Understanding → parse row/column/cell → trích xuất value, unit, year, Scope
```

#### 7. `table_text` — Tier 3

`table_text` dùng cho các vùng có dạng bảng nhưng chủ yếu là chữ, không có số liệu, hoặc chỉ có một cột số liệu nhưng không phải thông tin cần thiết, ví dụ số trang. Đây không phải là thông tin bổ sung cho bảng số mà là bảng toàn chữ, ít có ý nghĩa về số liệu Carbon.

Rule đánh `table_text`:

```text
- Có cấu trúc giống bảng
- Chủ yếu chứa text
- Không có số liệu quan trọng
- Có thể chỉ có một cột số liệu nhưng không phục vụ mục tiêu extraction
- Ví dụ: bảng mục, bảng liệt kê, bảng chứa số trang, bảng toàn chữ
```

#### 8. `text` — Tier 2

`text` dùng cho một cụm văn bản chính. Có thể đóng khung cả tên mục và nội dung của mục đó nếu chúng thuộc cùng một cụm nội dung.

Vai trò sau LayoutLMv3:

```text
text → Semantic Mapping / RAG → tìm thông tin ESG, Scope, carbon-related evidence
```

#### 9. `toc` — Tier 4

`toc` là Table of Content, tức mục lục hoặc bảng nội dung. Label này dùng cho các vùng liệt kê cấu trúc báo cáo, danh sách chương/mục và số trang tương ứng.

Ví dụ nên đánh `toc`:

```text
- Mục lục báo cáo
- Danh sách nội dung
- Các dòng liệt kê chương/mục kèm số trang
```

Label mapping sử dụng trong code:

```python
labels = [
    "O",
    "chart",
    "figure",
    "footer",
    "header",
    "ignore",
    "table",
    "table_text",
    "text",
    "toc"
]

label2id = {
    "O": 0,
    "chart": 1,
    "figure": 2,
    "footer": 3,
    "header": 4,
    "ignore": 5,
    "table": 6,
    "table_text": 7,
    "text": 8,
    "toc": 9
}
```

Mapping được lưu tại:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/label_mapping/label2id.json
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/label_mapping/id2label.json
```

Kết quả kiểm tra label:

```text
Unknown labels: set()
Labels not appearing: set()
```

Điều này nghĩa là:

```text
- Không có label lạ ngoài 10 label đã khai báo.
- Tất cả 10 label đều xuất hiện trong dataset.
```

---

## 4. Kiểm tra dataset trước khi train

Kết quả kiểm tra JSON và ảnh:

```text
Number of pages: 2466
File name: bao_viet_holdings_2024_p345_jpg.rf.a30d88b69fd1694f4aa1e448d0835a10.jpg
Words: 44
Bboxes: 44
Labels: 44
Image exists: True
```

Kiểm tra toàn bộ dataset:

```text
Missing images: 0
Length error: 0
Bbox error: 0
```

Ý nghĩa:

| Chỉ số | Kết quả | Ý nghĩa |
|---|---:|---|
| `Missing images` | 0 | Tất cả `file_name` trong JSON đều có ảnh tương ứng |
| `Length error` | 0 | `words`, `bboxes`, `labels` luôn có cùng độ dài |
| `Bbox error` | 0 | Tất cả bbox hợp lệ và nằm trong thang 0–1000 |

Dataset đã đạt điều kiện để train LayoutLMv3.

---

## 5. Phân bố label trong dataset

Tổng số OCR word/line tokens: **179,040**.

| Label | Số token |
|---|---:|
| `text` | 68,357 |
| `table` | 39,889 |
| `toc` | 19,656 |
| `table_text` | 13,418 |
| `ignore` | 12,777 |
| `figure` | 12,095 |
| `chart` | 5,649 |
| `header` | 5,060 |
| `footer` | 2,051 |
| `O` | 88 |

Nhận xét:

- Dataset bị mất cân bằng label, trong đó `text` và `table` chiếm nhiều token nhất.
- Các label quan trọng cho pipeline downstream là `table`, `table_text`, `chart`, `text`.
- Label `O` rất ít, chỉ có 88 token. Đây là token không thuộc region annotation nào, không phải 88 ảnh.
- Vì label bị mất cân bằng, khi đánh giá mô hình không nên chỉ nhìn accuracy hoặc weighted F1; cần xem thêm macro F1 và F1-score theo từng label.

---

## 6. Split train / validation / test

Dataset được split theo tỷ lệ:

```text
80% train
10% validation
10% test
```

Code split:

```python
from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    shuffle=True
)
```

Kết quả:

| Split | Số page | Vai trò |
|---|---:|---|
| Train | 1972 | Dùng để cập nhật trọng số model |
| Validation | 247 | Dùng để theo dõi `eval_loss` và chọn checkpoint tốt nhất |
| Test | 247 | Dùng để đánh giá cuối cùng sau train |

Các file split được lưu vào Drive:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/splits/train_data.json
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/splits/val_data.json
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/splits/test_data.json
```

---

## 7. Processor LayoutLMv3

Processor được load như sau:

```python
from transformers import LayoutLMv3Processor

processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)
```

Điểm quan trọng:

```text
apply_ocr=False
```

Lý do:

```text
Dataset đã có OCR words và OCR bboxes từ PaddleOCR.
Nếu để apply_ocr=True, processor sẽ OCR lại ảnh, tạo words/bboxes khác, làm sai alignment với labels trong JSON.
```

Processor nhận:

```text
image
words
boxes
word_labels
```

và tạo ra input tensor cho LayoutLMv3:

```text
input_ids
attention_mask
bbox
labels
pixel_values
```

---

## 8. Dataset class cho LayoutLMv3

Dataset class được xây dựng để mỗi sample trả về đúng input cho LayoutLMv3:

```python
class LayoutDataset(Dataset):
    def __init__(self, examples, image_dir, processor, label2id, max_length=512):
        self.examples = examples
        self.image_dir = image_dir
        self.processor = processor
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        image_path = os.path.join(self.image_dir, item["file_name"])
        image = Image.open(image_path).convert("RGB")

        words = item["words"]
        boxes = item["bboxes"]
        labels_item = item["labels"]

        word_labels = [self.label2id[label] for label in labels_item]

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding
```

Thông số:

```python
MAX_LENGTH = 512
```

Lý do dùng `max_length=512`:

```text
LayoutLMv3 là transformer-based model, sequence length phổ biến là 512.
Nếu page có nhiều token hơn 512, phần dư sẽ bị cắt bởi truncation=True.
Nếu ít hơn 512, processor padding lên đúng 512.
```

Shape của một sample sau processor:

```text
input_ids      torch.Size([512])
attention_mask torch.Size([512])
bbox           torch.Size([512, 4])
labels         torch.Size([512])
pixel_values   torch.Size([3, 224, 224])
```

Ý nghĩa:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `input_ids` | `[512]` | Token IDs sau tokenizer |
| `attention_mask` | `[512]` | Xác định token thật và padding |
| `bbox` | `[512, 4]` | Bbox của từng token theo thang 0–1000 |
| `labels` | `[512]` | Label ID của từng token; padding/special token là `-100` |
| `pixel_values` | `[3, 224, 224]` | Ảnh page đã được resize/normalize cho LayoutLMv3 |

---

## 9. Model được fine-tune

Model sử dụng:

```python
from transformers import LayoutLMv3ForTokenClassification

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
```

Thông tin chính:

| Thành phần | Giá trị |
|---|---|
| Base model | `microsoft/layoutlmv3-base` |
| Task head | Token classification |
| Số label | 10 |
| Input | Image + OCR words + bbox |
| Output | Label cho từng token |

Khi load model, classifier head được khởi tạo cho 10 label của dataset. Đây là hành vi bình thường vì head của bài toán token classification được fine-tune lại theo label riêng.

---

## 10. Cấu hình train

Checkpoint được lưu tại:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/layoutlmv3_checkpoints
```

TrainingArguments:

```python
training_args = TrainingArguments(
    output_dir=DRIVE_OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=5e-5,
    num_train_epochs=10,
    weight_decay=0.01,

    logging_steps=50,

    eval_strategy="steps",
    eval_steps=250,

    save_strategy="steps",
    save_steps=250,

    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=True,
    report_to="none",

    remove_unused_columns=False
)
```

Giải thích các cấu hình quan trọng:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `per_device_train_batch_size` | 1 | Mỗi lần GPU xử lý 1 page để tránh OOM |
| `gradient_accumulation_steps` | 8 | Tạo effective batch size = 8 |
| `learning_rate` | 5e-5 | Learning rate dùng cho fine-tuning Transformer |
| `num_train_epochs` | 10 | Train tối đa 10 epoch |
| `eval_steps` | 250 | Đánh giá validation gần mỗi epoch |
| `save_steps` | 250 | Lưu checkpoint gần mỗi epoch |
| `save_total_limit` | 3 | Giữ tối đa 3 checkpoint để tránh đầy Drive |
| `load_best_model_at_end` | True | Sau train, tự load checkpoint tốt nhất |
| `metric_for_best_model` | `eval_loss` | Chọn checkpoint theo validation loss |
| `greater_is_better` | False | Vì eval loss càng thấp càng tốt |
| `fp16` | True | Mixed precision trên GPU T4 |
| `remove_unused_columns` | False | Giữ lại các field đặc biệt như `bbox`, `pixel_values` |

Trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor
)
```

---

## 11. Log train và dấu hiệu overfitting

Quá trình train chạy đủ 10 epoch:

```text
2470 / 2470 steps
Epoch 10 / 10
```

Log loss:

| Step | Training Loss | Validation Loss |
|---:|---:|---:|
| 250 | 4.176415 | 0.473471 |
| 500 | 3.117473 | 0.445299 |
| 750 | 2.847447 | 0.423347 |
| 1000 | 1.882460 | 0.366579 |
| 1250 | 1.479305 | 0.371066 |
| 1500 | 1.006185 | 0.429786 |
| 1750 | 0.820629 | 0.434348 |
| 2000 | 0.460363 | 0.442501 |
| 2250 | 0.261082 | 0.459352 |
| 2470 | 0.277001 | 0.468479 |

Nhận xét:

```text
- Training loss giảm liên tục, cho thấy model học tốt trên train set.
- Validation loss giảm đến step 1000 rồi tăng trở lại.
- Từ sau step 1000 có dấu hiệu overfitting.
- Vì load_best_model_at_end=True, model cuối cùng được Trainer load lại là checkpoint có validation loss thấp nhất, không phải checkpoint cuối.
```

Best checkpoint:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/layoutlmv3_checkpoints/checkpoint-1000
```

Best validation loss:

```text
0.3665787875652313
```

Kết luận về train:

```text
Train đã hoàn tất.
Model tốt nhất là checkpoint-1000.
Checkpoint cuối không phải model tốt nhất vì validation loss tăng ở các epoch sau.
```

---

## 12. Đánh giá trên test set

Sau khi train xong, chạy:

```python
test_result = trainer.evaluate(eval_dataset=test_dataset)
```

Sau đó chạy prediction:

```python
predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels_true = predictions.label_ids
pred_ids = np.argmax(logits, axis=-1)
```

Khi tính metric, các token có label `-100` được bỏ qua:

```python
if label_id != -100:
    true_labels.append(id2label[int(label_id)])
    pred_labels.append(id2label[int(pred_id)])
```

Số token được đánh giá:

```text
Number of evaluated tokens: 14583
```

---

## 13. Classification report

Kết quả đánh giá theo từng label:

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `O` | 0.0000 | 0.0000 | 0.0000 | 4 |
| `chart` | 0.9330 | 0.8813 | 0.9064 | 379 |
| `figure` | 0.7418 | 0.7229 | 0.7322 | 1061 |
| `footer` | 0.8304 | 0.5254 | 0.6436 | 177 |
| `header` | 0.8801 | 0.8633 | 0.8716 | 578 |
| `ignore` | 0.8427 | 0.7184 | 0.7756 | 902 |
| `table` | 0.9465 | 0.9676 | 0.9569 | 3364 |
| `table_text` | 0.8344 | 0.9108 | 0.8709 | 863 |
| `text` | 0.9239 | 0.9480 | 0.9358 | 5380 |
| `toc` | 0.9940 | 0.9765 | 0.9852 | 1875 |

Overall:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.9129 |
| Macro Precision | 0.7927 |
| Macro Recall | 0.7514 |
| Macro F1 | 0.7678 |
| Weighted Precision | 0.9117 |
| Weighted Recall | 0.9129 |
| Weighted F1 | 0.9113 |

Micro metrics:

| Metric | Giá trị |
|---|---:|
| Micro Precision | 0.9129 |
| Micro Recall | 0.9129 |
| Micro F1 | 0.9129 |

---

## 14. Đánh giá kết quả mô hình

### 14.1. Nhóm label đạt rất tốt

| Label | F1-score | Nhận xét |
|---|---:|---|
| `toc` | 0.9852 | Mục lục được nhận diện rất tốt |
| `table` | 0.9569 | Vùng bảng được nhận diện rất tốt |
| `text` | 0.9358 | Text chính được nhận diện tốt |
| `chart` | 0.9064 | Chart được nhận diện tốt, đủ dùng cho bước Chart Understanding |
| `header` | 0.8716 | Header nhận diện khá tốt |
| `table_text` | 0.8709 | Text liên quan bảng nhận diện khá tốt |

Đây là nhóm label quan trọng nhất cho pipeline:

```text
table / table_text → Table Understanding
chart → Chart Understanding
text → Semantic Mapping / RAG
```

### 14.2. Nhóm label trung bình

| Label | F1-score | Nhận xét |
|---|---:|---|
| `ignore` | 0.7756 | Dùng được, nhưng còn khả năng nhầm với text/figure |
| `figure` | 0.7322 | Trung bình khá, do figure có nhiều dạng khác nhau |
| `footer` | 0.6436 | Yếu hơn các label khác, có thể do ít dữ liệu và vị trí/kiểu footer không đồng nhất |

### 14.3. Label `O`

| Label | F1-score | Support | Nhận xét |
|---|---:|---:|---|
| `O` | 0.0000 | 4 | Không đáng lo vì test set chỉ có 4 token `O` |

Label `O` không có nhiều ý nghĩa trong pipeline chính vì nó chỉ đại diện cho token không nằm trong vùng annotation nào. Việc F1 của `O` bằng 0 không ảnh hưởng lớn đến mục tiêu chính.

---

## 15. Mô hình sau train có thể làm gì

Model LayoutLMv3 đã fine-tune có thể nhận input:

```text
Ảnh page sạch
+ OCR words
+ OCR bboxes normalized 0–1000
```

và predict label layout cho từng OCR word/token:

```text
O, chart, figure, footer, header, ignore, table, table_text, text, toc
```

Output có thể dùng để:

1. **Nhận diện vùng bảng**
   ```text
   pred_label == table hoặc table_text
   ```
   Sau đó gom vùng này để đưa sang Table Understanding.

2. **Nhận diện vùng biểu đồ**
   ```text
   pred_label == chart
   ```
   Sau đó đưa sang Chart Understanding.

3. **Nhận diện text chính**
   ```text
   pred_label == text
   ```
   Dùng cho semantic extraction, Scope Mapping và RAG.

4. **Loại bỏ nhiễu**
   ```text
   pred_label in header/footer/ignore
   ```
   Có thể loại hoặc xử lý riêng.

5. **Hỗ trợ điều hướng tài liệu**
   ```text
   pred_label == toc
   ```
   Giúp nhận diện mục lục và cấu trúc báo cáo.

---

## 16. Mô hình chưa làm được gì

Mô hình LayoutLMv3 hiện tại **chưa trực tiếp trích xuất Carbon footprint**.

Nó chưa thực hiện:

```text
- Parse cấu trúc bảng row/column/cell.
- Gán giá trị phát thải vào Scope 1/2/3.
- Chuẩn hóa đơn vị tCO2e, kgCO2e.
- Trích xuất giá trị từ biểu đồ.
- Đối chiếu GHG Protocol hoặc GRI Standards bằng RAG.
- Giải thích kết quả bằng XAI.
- Phát hiện greenwashing.
```

Vai trò của nó là:

```text
Layout Understanding / Word-level Region Classification
```

Nó là bước tiền xử lý thông minh để xác định vùng nào cần chuyển sang các module sau.

---

## 17. Kết luận

Mô hình `LayoutLMv3ForTokenClassification` đã được fine-tune thành công trên dataset ESG / Annual Report đã merge OCR với region label.

Kết quả chính:

```text
Best checkpoint: checkpoint-1000
Best validation loss: 0.3665787875652313
Test accuracy: 0.9129
Weighted F1: 0.9113
Macro F1: 0.7678
```

Các label chính cho downstream đạt tốt:

```text
table F1 = 0.9569
table_text F1 = 0.8709
chart F1 = 0.9064
text F1 = 0.9358
```

Kết luận thực tế:

```text
Model đủ tốt để dùng làm Layout Understanding baseline cho đồ án.
Có thể chuyển sang bước inference thử trên ảnh mới, visualize prediction, sau đó gom vùng table/chart/text để nối sang Table Understanding, Chart Understanding, Semantic Mapping và RAG.
```

---

## 18. Các đường dẫn liên quan

Dataset:

```text
/content/layoutlmv3_data/valid/
```

Ảnh train:

```text
/content/layoutlmv3_data/valid/images
```

JSON train:

```text
/content/layoutlmv3_data/valid/0_layoutlmv3_dataset_normalized.json
```

Split:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/splits/train_data.json
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/splits/val_data.json
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/splits/test_data.json
```

Label mapping:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/label_mapping/label2id.json
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/label_mapping/id2label.json
```

Checkpoint:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/layoutlmv3_checkpoints
```

Best checkpoint:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/layoutlmv3_checkpoints/checkpoint-1000
```

Khuyến nghị nơi lưu final model:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/layoutlmv3_finetuned_final
```

Khuyến nghị nơi lưu final model zip:

```text
/content/drive/MyDrive/Doan/Dataset_Layoutlmv3/layoutlmv3_finetuned_final.zip
```

Notebook train:

```text
TrainLayoutlmv3.ipynb
```
