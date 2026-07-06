# OCR Preparation Pipeline for LayoutLMv3

## Mục tiêu

Chuyển đổi mỗi trang PDF đã render thành một mẫu dữ liệu hoàn chỉnh để
fine-tune LayoutLMv3. Pipeline kết hợp OCR ở mức **word-level** với
annotation vùng (region annotation) nhằm tạo ra nhãn cho từng từ
(token-level labels).

------------------------------------------------------------------------

# Bước 1. Kiểm tra tính đồng nhất của ảnh và annotation

## Mục đích

Đảm bảo ảnh dùng để OCR và ảnh đã annotate trên Roboflow sử dụng cùng hệ
tọa độ.

Nếu ảnh OCR và ảnh annotation khác kích thước hoặc khác tên file thì
bounding box sẽ không còn khớp và không thể merge trực tiếp.

## Kiểm tra

-   Tên file
-   Chiều rộng
-   Chiều cao

giữa:

-   Ảnh thực tế
-   `_annotations.coco.json`

Nếu có sai lệch:

-   OCR phải chạy trên chính ảnh đã annotate; hoặc
-   Rescale lại annotation.

Script sử dụng:

`check_image_alignment.py`

Kết quả mong muốn:

    Image Coordinate System
    ==
    Annotation Coordinate System

------------------------------------------------------------------------

# Bước 2. Word-level OCR

## Input

    page_001.png

## Xử lý

Sử dụng PaddleOCR để nhận dạng văn bản ở mức **word-level**.

Mỗi từ bao gồm:

-   Text
-   Confidence
-   Bounding Box

PaddleOCR trả về polygon 4 điểm, sau đó chuyển thành bounding box chuẩn:

    [x0, y0, x1, y1]

Ví dụ:

``` json
{
    "text":"Scope",
    "bbox":[120,145,190,170],
    "conf":0.998
}
```

Kết quả toàn bộ dataset được lưu thành:

    ocr_words.json

Lúc này dữ liệu chỉ gồm:

-   Word
-   Bounding Box
-   Confidence

Chưa có label.

------------------------------------------------------------------------

# Bước 3. Region Annotation

Song song với OCR, Roboflow cung cấp:

    _annotations.coco.json

Mỗi annotation gồm:

-   Category
-   Bounding Box

Ví dụ:

``` json
{
    "category":"table",
    "bbox":[100,80,700,900]
}
```

Đây là annotation ở mức Region-level.

------------------------------------------------------------------------

# Bước 4. Merge OCR với Region Annotation

## Mục tiêu

Chuyển:

    Region-level Annotation

thành

    Word-level Annotation

## Thuật toán

Đối với từng word:

1.  Tính tâm của bounding box:

```{=html}
<!-- -->
```
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

2.  Kiểm tra tâm nằm trong region nào.

Ví dụ:

    Table
    bbox = [100,80,700,900]

    ↓

    Word "Scope"

    ↓

    Label = table

Nếu một word nằm trong nhiều vùng:

-   Ưu tiên vùng có diện tích nhỏ hơn.

Ví dụ:

    Table
    └── Table_text

Word sẽ nhận:

    table_text

Nếu không thuộc vùng nào:

    Label = O

Ví dụ output:

``` json
{
    "text":"Scope",
    "bbox":[120,145,190,170],
    "label":"table"
}
```

Script:

    merge_json.py

------------------------------------------------------------------------

# Bước 5. Tạo LayoutLMv3 Dataset

Sau khi merge, dữ liệu được gom theo từng trang.

Ví dụ:

``` json
{
    "image_id":1,
    "file_name":"page_001.png",

    "words":[
        "Scope",
        "1",
        "Emission"
    ],

    "bboxes":[
        [...],
        [...],
        [...]
    ],

    "labels":[
        "table",
        "table",
        "table"
    ]
}
```

File kết quả:

    layoutlmv3_dataset.json

------------------------------------------------------------------------

# Bước 6. Dataset Quality Verification

Mục đích:

Kiểm tra merge có chính xác hay không.

Script:

    visualize.py

Script sẽ hiển thị:

-   OCR Bounding Box
-   OCR Text
-   Assigned Label

Qua đó có thể phát hiện:

-   OCR sai
-   Label sai
-   Merge sai
-   Bounding Box sai

trước khi train.

------------------------------------------------------------------------

# Bước 7. Bounding Box Normalization

LayoutLMv3 không sử dụng tọa độ pixel gốc.

Ví dụ:

Ảnh:

    2480 × 3508

OCR:

    [620,700,900,760]

Sau normalize:

    [250,199,362,216]

Công thức:

    x' = x / image_width × 1000
    y' = y / image_height × 1000

Sau bước này mọi bounding box đều nằm trong khoảng:

    0 → 1000

Đây là định dạng LayoutLMv3 yêu cầu.

------------------------------------------------------------------------

# Bước 8. LayoutLMv3 Processor

Sau khi normalize, dữ liệu được đưa vào `LayoutLMv3Processor`.

Processor kết hợp đồng thời:

-   Page Image
-   Tokens
-   Bounding Boxes
-   Labels

để tạo tensor huấn luyện.

Output:

    input_ids
    attention_mask
    bbox
    pixel_values
    labels

Đây là đầu vào chuẩn để fine-tune LayoutLMv3.

------------------------------------------------------------------------

# Output cuối cùng

Mỗi trang PDF trở thành một sample hoàn chỉnh.

Ví dụ:

``` json
{
    "image":"page_001.png",

    "tokens":[
        "Scope",
        "1",
        "Emission",
        "12,441"
    ],

    "bbox":[
        [250,198,290,214],
        [292,198,298,214],
        [300,198,380,214],
        [390,198,430,214]
    ],

    "labels":[
        "table",
        "table",
        "table",
        "table"
    ]
}
```

------------------------------------------------------------------------

# Pipeline tổng thể

``` text
PDF Pages
      │
      ▼
Rendered Images
      │
      ▼
Image Alignment Verification
      │
      ▼
PaddleOCR (Word-level)
      │
      ▼
OCR JSON
      │
      ▼
COCO Region Annotation
      │
      ▼
Merge OCR + Region Annotation
      │
      ▼
Word-level Labels
      │
      ▼
Visualization & QA
      │
      ▼
Bounding Box Normalization (0–1000)
      │
      ▼
LayoutLMv3 Processor
      │
      ▼
Fine-tuning Dataset
```
