# Kế hoạch triển khai Table Understanding, Chart Understanding và Semantic Mapping

**Bối cảnh:** Tài liệu này mô tả kế hoạch triển khai các bước tiếp theo sau khi đã fine-tune xong LayoutLMv3 cho bài toán layout region classification. Nội dung chỉ tập trung vào **luồng xử lý**, không trình bày lựa chọn mô hình cụ thể.

---

## 1. Vị trí của ba bước trong pipeline tổng thể

Sau khi OCR và LayoutLMv3 đã hoàn thành, hệ thống có khả năng nhận diện từng OCR word/token thuộc các vùng như `table`, `table_text`, `chart`, `text`, `figure`, `header`, `footer`, `toc`, `ignore` hoặc `O`.

Ba bước tiếp theo nằm trong Module 1 như sau:

```text
PDF / ESG Report / Annual Report
        ↓
Render PDF thành ảnh từng trang
        ↓
OCR lấy words + bbox
        ↓
LayoutLMv3 dự đoán layout label từng OCR word
        ↓
Table Understanding
        ↓
Chart Understanding
        ↓
Semantic Mapping Scope 1 / Scope 2 / Scope 3
        ↓
Structured ESG Output
```

Logic ưu tiên dữ liệu:

```text
Table data > Chart data > Text evidence
```

Ý nghĩa:

- Nếu có bảng chứa số liệu phát thải, ưu tiên lấy dữ liệu từ bảng.
- Nếu không có bảng rõ ràng nhưng có biểu đồ liên quan, dùng Chart Understanding để trích xuất thông tin.
- Nếu bảng và biểu đồ không đủ thông tin, dùng text evidence để hỗ trợ Semantic Mapping.

---

## 2. Input đầu vào sau LayoutLMv3

Sau bước LayoutLMv3 inference, mỗi trang nên có output dạng layout-aware JSON.

Ví dụ:

```json
{
  "file_name": "company_2024_p024.jpg",
  "page_index": 24,
  "width": 2382,
  "height": 3367,
  "tokens": [
    {
      "word": "Scope 1",
      "bbox": [120, 240, 210, 260],
      "bbox_pixel": [285, 808, 500, 875],
      "ocr_conf": 0.98,
      "pred_label": "table_text"
    },
    {
      "word": "12,441",
      "bbox": [230, 240, 300, 260],
      "bbox_pixel": [548, 808, 714, 875],
      "ocr_conf": 0.97,
      "pred_label": "table"
    }
  ]
}
```

Các label được dùng trực tiếp cho ba bước tiếp theo:

| Label | Vai trò sau LayoutLMv3 |
|---|---|
| `table` | Dữ liệu bảng có số liệu, ưu tiên cao nhất cho carbon extraction |
| `table_text` | Bảng dạng chữ hoặc vùng text có cấu trúc bảng, dùng làm context phụ |
| `chart` | Vùng biểu đồ, dùng cho Chart Understanding |
| `text` | Đoạn văn ESG, dùng cho Semantic Mapping và evidence |
| `figure` | Hình có chữ và có ý nghĩa, có thể dùng làm evidence phụ nếu liên quan ESG |
| `header`, `footer`, `ignore`, `toc`, `O` | Thường không dùng làm nguồn số liệu chính, chủ yếu lọc nhiễu hoặc hỗ trợ điều hướng |

---

## 3. Table Understanding

### 3.1. Mục tiêu

Table Understanding có nhiệm vụ chuyển vùng bảng từ layout-level token prediction thành dữ liệu bảng có cấu trúc.

Mục tiêu cụ thể:

```text
Vùng table/table_text
→ xác định bảng hoàn chỉnh
→ khôi phục hàng/cột/cell
→ đọc header, unit, year, scope, value
→ chuẩn hóa thành structured table records
```

Ví dụ output mong muốn:

```json
{
  "source_type": "table",
  "page_index": 24,
  "table_id": "p024_table_01",
  "records": [
    {
      "row_label": "Scope 1 emissions",
      "column_label": "2024",
      "value": "12441",
      "unit": "tCO2e",
      "raw_text": "Scope 1 emissions 12,441 tCO2e",
      "bbox": [100, 220, 600, 420]
    }
  ]
}
```

---

### 3.2. Luồng xử lý Table Understanding

```text
LayoutLMv3 output
        ↓
Lọc token có label table và table_text
        ↓
Gom token theo page
        ↓
Xác định các cụm bảng trên từng page
        ↓
Tạo table region bbox
        ↓
Crop vùng bảng từ ảnh gốc nếu cần
        ↓
Phân tích cấu trúc bảng: row / column / cell
        ↓
Gán text vào từng cell
        ↓
Xác định header, unit, year, metric name, value
        ↓
Chuẩn hóa record bảng
        ↓
Đưa sang Semantic Mapping
```

---

### 3.3. Bước 1: Lọc token bảng

Từ output LayoutLMv3, lấy các token có label:

```text
table
table_text
```

Quy tắc:

- `table` được ưu tiên vì đây là bảng hoàn chỉnh có số liệu.
- `table_text` dùng làm context phụ hoặc dùng riêng nếu không có bảng số liệu.
- Nếu `table` và `table_text` nằm gần nhau trong cùng vùng, có thể gom thành một table candidate.

Output bước này:

```json
{
  "page_index": 24,
  "table_tokens": [...],
  "table_text_tokens": [...]
}
```

---

### 3.4. Bước 2: Gom token thành table region

Do LayoutLMv3 dự đoán theo từng word/token, cần gom các token gần nhau thành vùng bảng.

Cách gom:

1. Gom theo page.
2. Sắp xếp token theo tọa độ `y`, sau đó `x`.
3. Nhóm token gần nhau theo khoảng cách không gian.
4. Tạo bbox bao ngoài cho từng cụm token.
5. Mỗi cụm tương ứng một table candidate.

Output:

```json
{
  "table_id": "p024_table_01",
  "page_index": 24,
  "bbox_pixel": [x1, y1, x2, y2],
  "tokens": [...]
}
```

Lưu ý:

- Một page có thể có nhiều bảng.
- Không nên gom nhiều hơn một bảng độc lập vào cùng một table region.
- Nếu hai bảng gần nhau nhưng có tiêu đề/khối trắng tách biệt, nên tách riêng.

---

### 3.5. Bước 3: Khôi phục cấu trúc hàng và cột

Sau khi có table region, cần xác định cấu trúc bảng.

Luồng xử lý:

```text
Table region tokens
        ↓
Sắp xếp token theo vị trí
        ↓
Nhóm token theo dòng dựa trên tọa độ y
        ↓
Nhóm token theo cột dựa trên tọa độ x
        ↓
Tạo cell candidates
        ↓
Gán text vào cell
```

Kết quả trung gian:

```json
{
  "table_id": "p024_table_01",
  "rows": [
    {
      "row_index": 0,
      "cells": [
        {"col_index": 0, "text": "Emission category"},
        {"col_index": 1, "text": "2023"},
        {"col_index": 2, "text": "2024"}
      ]
    },
    {
      "row_index": 1,
      "cells": [
        {"col_index": 0, "text": "Scope 1"},
        {"col_index": 1, "text": "10,250"},
        {"col_index": 2, "text": "12,441"}
      ]
    }
  ]
}
```

---

### 3.6. Bước 4: Nhận diện thành phần trong bảng

Cần xác định các thành phần quan trọng:

| Thành phần | Ý nghĩa |
|---|---|
| Metric name | Tên chỉ số, ví dụ `Scope 1 emissions`, `GHG emissions`, `Energy consumption` |
| Value | Giá trị số, ví dụ `12,441`, `5.2`, `100%` |
| Unit | Đơn vị, ví dụ `tCO2e`, `tons CO2e`, `MWh`, `GJ` |
| Year / Period | Năm hoặc kỳ báo cáo, ví dụ `2023`, `2024`, `FY2024` |
| Scope hint | Tín hiệu gợi ý Scope 1/2/3 |
| Footnote | Ghi chú, có thể ảnh hưởng đến cách hiểu dữ liệu |

Output:

```json
{
  "metric_name": "Scope 1 emissions",
  "value": "12,441",
  "unit": "tCO2e",
  "year": "2024",
  "scope_hint": "Scope 1",
  "source_type": "table",
  "source_page": 24,
  "source_bbox": [x1, y1, x2, y2]
}
```

---

### 3.7. Bước 5: Chuẩn hóa dữ liệu bảng

Các giá trị trong bảng cần được chuẩn hóa trước khi đưa sang Semantic Mapping.

Cần chuẩn hóa:

```text
12,441 → 12441
12.441 nếu là định dạng Việt Nam → cần kiểm tra ngữ cảnh
5.2 million → 5200000
% → percentage
FY2024 → 2024
tCO₂e / tCO2e / tons CO2-e → tCO2e
```

Output sau chuẩn hóa:

```json
{
  "source_type": "table",
  "metric_text": "Scope 1 emissions",
  "value": 12441,
  "unit": "tCO2e",
  "year": 2024,
  "page_index": 24,
  "confidence_signals": {
    "has_numeric_value": true,
    "has_unit": true,
    "has_year": true,
    "has_scope_keyword": true
  }
}
```

---

## 4. Chart Understanding

### 4.1. Mục tiêu

Chart Understanding có nhiệm vụ xử lý các vùng được LayoutLMv3 dự đoán là `chart`, nhằm trích xuất thông tin từ biểu đồ.

Mục tiêu:

```text
Vùng chart
→ xác định chart region
→ đọc text trong chart
→ nhận diện trục, nhãn, legend, đơn vị, năm, giá trị
→ trích xuất dữ liệu hoặc xu hướng
→ chuẩn hóa thành chart records
```

Chart không phải nguồn chính nếu có bảng số liệu rõ ràng. Chart được dùng khi:

- Không có bảng tương ứng.
- Bảng thiếu dữ liệu nhưng chart có xu hướng hoặc giá trị.
- Cần đối chiếu narrative với số liệu trực quan.
- Cần lấy trend phát thải qua nhiều năm.

---

### 4.2. Luồng xử lý Chart Understanding

```text
LayoutLMv3 output
        ↓
Lọc token có label chart
        ↓
Gom token thành chart region theo page
        ↓
Crop vùng chart từ ảnh gốc
        ↓
OCR lại hoặc dùng OCR token có sẵn trong vùng chart
        ↓
Nhận diện thành phần chart: title / axis / legend / label / value
        ↓
Trích xuất datapoint hoặc trend
        ↓
Chuẩn hóa dữ liệu chart
        ↓
Đưa sang Semantic Mapping
```

---

### 4.3. Bước 1: Lọc và gom chart region

Từ output LayoutLMv3, lấy token có label:

```text
chart
```

Sau đó:

1. Gom theo page.
2. Gom các token `chart` gần nhau thành chart candidate.
3. Tạo bbox bao ngoài vùng chart.
4. Nếu một page có nhiều chart nhỏ, tách từng chart riêng nếu khoảng cách đủ rõ.
5. Không gom quá nhiều biểu đồ độc lập vào cùng một vùng.

Output:

```json
{
  "chart_id": "p030_chart_01",
  "page_index": 30,
  "bbox_pixel": [x1, y1, x2, y2],
  "tokens": [...]
}
```

---

### 4.4. Bước 2: Xác định thành phần trong chart

Các thành phần cần tìm trong chart:

| Thành phần | Vai trò |
|---|---|
| Chart title | Cho biết biểu đồ đang nói về chỉ số nào |
| X-axis label | Thường là năm, tháng, nhóm phân loại hoặc Scope |
| Y-axis label | Thường là giá trị và đơn vị |
| Legend | Phân biệt các series như Scope 1, Scope 2, Scope 3 |
| Data labels | Giá trị số được ghi trực tiếp trên cột/đường |
| Unit | tCO2e, MWh, GJ, %, VND, USD... |
| Trend direction | Tăng, giảm, ổn định |

Output trung gian:

```json
{
  "chart_id": "p030_chart_01",
  "title": "GHG emissions by scope",
  "x_axis": ["2022", "2023", "2024"],
  "legend": ["Scope 1", "Scope 2"],
  "unit": "tCO2e",
  "detected_values": ["1200", "1350", "1280"]
}
```

---

### 4.5. Bước 3: Trích xuất dữ liệu từ chart

Có hai loại output chart cần hỗ trợ.

#### Trường hợp 1: Chart có data labels rõ

Nếu chart có số ghi trực tiếp, lấy số đó làm giá trị.

Output:

```json
{
  "source_type": "chart",
  "metric_text": "GHG emissions by scope",
  "series": "Scope 1",
  "year": 2024,
  "value": 1280,
  "unit": "tCO2e",
  "page_index": 30
}
```

#### Trường hợp 2: Chart không có data labels rõ

Nếu chart không có số cụ thể, chỉ lấy thông tin xu hướng.

Output:

```json
{
  "source_type": "chart",
  "metric_text": "GHG emissions trend",
  "trend": "decreasing",
  "period": "2022-2024",
  "unit": "tCO2e",
  "page_index": 30,
  "value_status": "estimated_or_unavailable"
}
```

---

### 4.6. Bước 4: Chuẩn hóa dữ liệu chart

Dữ liệu từ chart cần được chuẩn hóa giống bảng.

Cần chuẩn hóa:

```text
Năm / period
Metric name
Series name
Value
Unit
Scope hint
Confidence
```

Output chuẩn hóa:

```json
{
  "source_type": "chart",
  "metric_text": "GHG emissions by scope",
  "scope_hint": "Scope 2",
  "value": 5221,
  "unit": "tCO2e",
  "year": 2024,
  "page_index": 30,
  "confidence_signals": {
    "has_chart_title": true,
    "has_data_label": true,
    "has_unit": true,
    "has_scope_keyword": true
  }
}
```

---

### 4.7. Quy tắc ưu tiên chart so với table

Nếu cùng một chỉ số xuất hiện ở cả bảng và chart:

```text
Ưu tiên bảng làm nguồn chính.
Chart dùng làm nguồn đối chiếu hoặc bổ sung xu hướng.
```

Ví dụ:

```json
{
  "metric": "Scope 2 emissions",
  "primary_source": "table",
  "supporting_source": "chart",
  "decision": "use_table_value"
}
```

Nếu chỉ có chart:

```json
{
  "metric": "Scope 2 emissions",
  "primary_source": "chart",
  "decision": "use_chart_value_or_trend"
}
```

---

## 5. Semantic Mapping

### 5.1. Mục tiêu

Semantic Mapping có nhiệm vụ ánh xạ dữ liệu đã trích xuất từ bảng, biểu đồ và text vào các nhóm ESG/carbon có ý nghĩa, đặc biệt là Scope 1, Scope 2 và Scope 3.

Input của Semantic Mapping không còn là OCR thô, mà là các record đã được chuẩn hóa từ Table Understanding, Chart Understanding và text evidence.

Mục tiêu:

```text
Structured records từ table/chart/text
→ hiểu ngữ nghĩa ESG
→ xác định Scope 1 / Scope 2 / Scope 3 hoặc carbon-related category
→ chuẩn hóa output cuối
```

---

### 5.2. Input của Semantic Mapping

Input gồm ba nhóm chính.

#### 5.2.1. Table records

```json
{
  "source_type": "table",
  "metric_text": "Purchased electricity emissions",
  "value": 5221,
  "unit": "tCO2e",
  "year": 2024,
  "page_index": 24
}
```

#### 5.2.2. Chart records

```json
{
  "source_type": "chart",
  "metric_text": "Indirect emissions from electricity consumption",
  "value": 5221,
  "unit": "tCO2e",
  "year": 2024,
  "page_index": 30
}
```

#### 5.2.3. Text evidence

```json
{
  "source_type": "text",
  "text": "The company reduced emissions from purchased electricity by increasing renewable energy usage.",
  "page_index": 28
}
```

---

### 5.3. Luồng xử lý Semantic Mapping

```text
Table records + Chart records + Text evidence
        ↓
Làm sạch text và chuẩn hóa thuật ngữ
        ↓
Nhận diện carbon-related candidates
        ↓
Trích xuất semantic signals
        ↓
Ánh xạ Scope 1 / Scope 2 / Scope 3
        ↓
Gán confidence và evidence
        ↓
Hợp nhất record trùng lặp
        ↓
Xuất structured ESG output
```

---

### 5.4. Bước 1: Làm sạch và chuẩn hóa thuật ngữ

Cần chuẩn hóa các biến thể ngôn ngữ ESG.

Ví dụ:

| Raw text | Normalized text |
|---|---|
| `GHG emission` | `greenhouse gas emissions` |
| `CO₂e` | `CO2e` |
| `tCO₂-e` | `tCO2e` |
| `electricity purchased` | `purchased electricity` |
| `fuel consumption` | `fuel combustion` |

Output:

```json
{
  "original_text": "Indirect emissions from electricity purchased",
  "normalized_text": "indirect emissions from purchased electricity"
}
```

---

### 5.5. Bước 2: Nhận diện carbon-related candidates

Không phải mọi dữ liệu trong table/chart/text đều liên quan đến carbon.

Cần lọc các candidate có tín hiệu liên quan:

```text
emission
carbon
CO2e
GHG
greenhouse gas
Scope 1
Scope 2
Scope 3
electricity
fuel
energy
logistics
business travel
supply chain
waste
```

Output:

```json
{
  "candidate_id": "cand_001",
  "is_carbon_related": true,
  "signals": ["emissions", "purchased electricity", "tCO2e"]
}
```

---

### 5.6. Bước 3: Trích xuất semantic signals

Với mỗi candidate, cần trích xuất các tín hiệu semantic để quyết định Scope.

| Signal | Ví dụ | Gợi ý Scope |
|---|---|---|
| Direct emissions | fuel combustion, company vehicles, boilers | Scope 1 |
| Purchased energy | purchased electricity, steam, heating, cooling | Scope 2 |
| Value chain | supplier, logistics, business travel, waste, purchased goods | Scope 3 |
| Explicit scope keyword | Scope 1 / Scope 2 / Scope 3 | Theo keyword |
| Unit | tCO2e, kgCO2e | Xác nhận đây là dữ liệu emission |
| Context | energy, operation, supply chain | Hỗ trợ phân loại |

Output:

```json
{
  "candidate_id": "cand_001",
  "semantic_signals": {
    "explicit_scope": null,
    "activity_type": "purchased electricity",
    "emission_type": "indirect",
    "unit": "tCO2e",
    "source_type": "table"
  }
}
```

---

### 5.7. Bước 4: Ánh xạ Scope

Quy tắc ánh xạ cơ bản:

```text
Nếu có explicit keyword Scope 1:
    map Scope 1
Nếu có explicit keyword Scope 2:
    map Scope 2
Nếu có explicit keyword Scope 3:
    map Scope 3
Nếu không có keyword rõ:
    dùng activity_type và context để suy luận
```

Ví dụ mapping:

| Activity / Text | Scope |
|---|---|
| Direct fuel combustion | Scope 1 |
| Company-owned vehicles | Scope 1 |
| Purchased electricity | Scope 2 |
| Purchased steam / heating / cooling | Scope 2 |
| Business travel | Scope 3 |
| Employee commuting | Scope 3 |
| Logistics / transportation by third party | Scope 3 |
| Purchased goods and services | Scope 3 |
| Waste generated in operations | Scope 3 |

Output:

```json
{
  "candidate_id": "cand_001",
  "mapped_scope": "Scope 2",
  "mapping_reason": "The record refers to indirect emissions from purchased electricity.",
  "confidence": 0.92
}
```

---

### 5.8. Bước 5: Hợp nhất record từ nhiều nguồn

Cùng một dữ liệu có thể xuất hiện ở nhiều nguồn:

```text
table
chart
text
```

Cần hợp nhất để tránh trùng lặp.

Quy tắc hợp nhất:

1. Nếu cùng metric, cùng year, cùng unit, cùng value → gộp thành một record.
2. Nếu table và chart khác nhau nhẹ → ưu tiên table, chart làm supporting evidence.
3. Nếu text chỉ mô tả, không có value → dùng làm explanation evidence.
4. Nếu value xung đột lớn → đánh dấu cần kiểm tra.

Ví dụ:

```json
{
  "company": "FPT Securities JSC",
  "year": 2024,
  "scope": "Scope 2",
  "emission_value": 5221,
  "unit": "tCO2e",
  "primary_source": {
    "source_type": "table",
    "page_index": 24
  },
  "supporting_sources": [
    {
      "source_type": "chart",
      "page_index": 30
    },
    {
      "source_type": "text",
      "page_index": 28
    }
  ]
}
```

---

### 5.9. Bước 6: Xuất structured ESG output

Output cuối của Semantic Mapping nên có dạng thống nhất:

```json
{
  "company": "FPT Securities JSC",
  "report_year": 2024,
  "scope": "Scope 2",
  "metric_name": "Purchased electricity emissions",
  "emission_value": 5221,
  "unit": "tCO2e",
  "source_type": "table",
  "source_page": 24,
  "source_bbox": [x1, y1, x2, y2],
  "confidence": 0.92,
  "evidence_text": "Purchased electricity emissions 5,221 tCO2e",
  "mapping_reason": "Purchased electricity is treated as indirect energy-related emissions."
}
```

---

## 6. Luồng kết hợp ba bước

Ba bước Table Understanding, Chart Understanding và Semantic Mapping nên được kết nối như sau:

```text
LayoutLMv3 output
        ↓
Tách nguồn dữ liệu theo label
        ↓
┌─────────────────────────────┐
│ table / table_text tokens   │
└──────────────┬──────────────┘
               ↓
        Table Understanding
               ↓
        Table records
               ↓
┌─────────────────────────────┐
│ chart tokens                │
└──────────────┬──────────────┘
               ↓
        Chart Understanding
               ↓
        Chart records
               ↓
┌─────────────────────────────┐
│ text tokens                 │
└──────────────┬──────────────┘
               ↓
        Text evidence records
               ↓
        Semantic Mapping
               ↓
        Scope 1 / Scope 2 / Scope 3 records
               ↓
        Structured ESG output
```

---

## 7. Output chuẩn trung gian cho toàn pipeline

Để các bước nối với nhau ổn định, nên thống nhất schema trung gian.

### 7.1. Region-level output

```json
{
  "region_id": "p024_table_01",
  "page_index": 24,
  "region_type": "table",
  "bbox_pixel": [x1, y1, x2, y2],
  "tokens": [...],
  "text": "..."
}
```

### 7.2. Extracted record output

```json
{
  "record_id": "rec_001",
  "source_type": "table",
  "metric_text": "Scope 1 emissions",
  "value": 12441,
  "unit": "tCO2e",
  "year": 2024,
  "page_index": 24,
  "bbox_pixel": [x1, y1, x2, y2]
}
```

### 7.3. Semantic mapped output

```json
{
  "record_id": "rec_001",
  "company": "FPT Securities JSC",
  "year": 2024,
  "scope": "Scope 1",
  "metric_name": "Scope 1 emissions",
  "value": 12441,
  "unit": "tCO2e",
  "source_type": "table",
  "source_page": 24,
  "confidence": 0.94,
  "evidence": [...]
}
```

---

## 8. Kiểm tra chất lượng từng bước

### 8.1. Kiểm tra Table Understanding

Cần kiểm tra:

```text
Có detect đúng vùng bảng không?
Có tách đúng từng bảng không?
Có giữ đúng row/column không?
Có lấy đúng value không?
Có lấy đúng unit/year không?
Có nhầm table_text thành table chính không?
```

Metric có thể dùng:

```text
Table region accuracy
Cell extraction accuracy
Value extraction accuracy
Unit extraction accuracy
Year extraction accuracy
Manual review trên một số page mẫu
```

---

### 8.2. Kiểm tra Chart Understanding

Cần kiểm tra:

```text
Có detect đúng chart region không?
Có tách đúng từng chart nhỏ không?
Có đọc đúng title/legend/axis không?
Có lấy được data label không?
Nếu không lấy được value, có lấy được trend không?
```

Metric có thể dùng:

```text
Chart region accuracy
Chart text extraction quality
Data label extraction accuracy
Trend classification accuracy
Manual review trên page có chart
```

---

### 8.3. Kiểm tra Semantic Mapping

Cần kiểm tra:

```text
Có nhận diện đúng record carbon-related không?
Có map đúng Scope 1/2/3 không?
Có bỏ qua dữ liệu không liên quan không?
Có gộp trùng đúng không?
Có giữ evidence và source page không?
```

Metric có thể dùng:

```text
Scope classification accuracy
Precision / Recall / F1 cho Scope 1/2/3
Carbon record extraction precision
Carbon record extraction recall
Manual validation theo từng báo cáo
```

---

## 9. Kế hoạch triển khai theo giai đoạn

### Giai đoạn 1: Chuẩn hóa output LayoutLMv3

Mục tiêu:

```text
Tạo layout-aware JSON cho từng page sau inference.
```

Cần làm:

1. Chạy OCR cho ảnh mới.
2. Normalize bbox 0–1000.
3. Dùng LayoutLMv3 predict label.
4. Lưu output gồm word, bbox, pred_label, page, file_name.
5. Visualize prediction để kiểm tra bằng mắt.

Output:

```text
layout_predictions.json
layout_predictions.csv
visualized_pages/
```

---

### Giai đoạn 2: Gom region table/chart/text

Mục tiêu:

```text
Chuyển word-level label thành region-level objects.
```

Cần làm:

1. Gom `table` và `table_text` thành table candidates.
2. Gom `chart` thành chart candidates.
3. Gom `text` thành text evidence blocks.
4. Loại bỏ `ignore`, `footer`, `header` nếu không cần.
5. Lưu region-level JSON.

Output:

```text
regions.json
```

---

### Giai đoạn 3: Table Understanding

Mục tiêu:

```text
Từ table regions tạo structured table records.
```

Cần làm:

1. Crop table region nếu cần.
2. Tách hàng/cột/cell.
3. Gán OCR text vào cell.
4. Xác định metric, value, unit, year.
5. Chuẩn hóa record.
6. Lưu table records.

Output:

```text
table_records.json
```

---

### Giai đoạn 4: Chart Understanding

Mục tiêu:

```text
Từ chart regions tạo chart records hoặc trend records.
```

Cần làm:

1. Crop chart region nếu cần.
2. Đọc title, axis, legend, label.
3. Trích xuất value nếu có data label.
4. Nếu không có value, trích xuất trend.
5. Chuẩn hóa chart record.
6. Lưu chart records.

Output:

```text
chart_records.json
```

---

### Giai đoạn 5: Semantic Mapping

Mục tiêu:

```text
Map table/chart/text records vào Scope 1/2/3.
```

Cần làm:

1. Gom table records, chart records, text records.
2. Lọc carbon-related candidates.
3. Trích xuất semantic signals.
4. Map Scope 1/2/3.
5. Gán confidence.
6. Gộp record trùng.
7. Xuất structured ESG output.

Output:

```text
structured_esg_output.json
```

---

## 10. Kết luận

Sau khi LayoutLMv3 đã nhận diện được các vùng `table`, `table_text`, `chart` và `text`, ba bước tiếp theo cần triển khai là Table Understanding, Chart Understanding và Semantic Mapping.

Luồng xử lý nên đi theo hướng:

```text
Word-level layout prediction
→ Region-level grouping
→ Source-specific understanding
→ Semantic mapping
→ Structured ESG output
```

Trong đó:

- Table Understanding là nguồn chính để lấy số liệu Carbon vì bảng có độ chính xác cao hơn.
- Chart Understanding là nguồn bổ sung để lấy giá trị hoặc xu hướng khi bảng không có hoặc thiếu dữ liệu.
- Semantic Mapping là bước biến dữ liệu thô đã trích xuất thành thông tin ESG có ý nghĩa, đặc biệt là Scope 1, Scope 2 và Scope 3.

Mục tiêu cuối cùng của ba bước này là tạo ra dữ liệu có cấu trúc, có nguồn trích dẫn rõ ràng, có page/bbox/evidence và sẵn sàng đưa sang RAG, XAI, Carbon analytics và greenwashing detection.
