# Data Pipeline Report — Eco-Lens

## 1. Data Description

### 1.1 Source

Dữ liệu được thu thập từ **587 báo cáo phát triển bền vững / báo cáo thường niên** của các doanh nghiệp Việt Nam. Quy trình thu thập được phân chia theo nhóm 4 thành viên dựa trên số trang của mỗi file PDF:

| Thành viên | Số trang được phân công |
|------------|------------------------|
| **Thống** | Page 1, 2 |
| **Bảo** | Page 3, 4, 5 |
| **Thư** | Page 6, 7, 8 |
| **Hào** | Page 9, 10, 11, 12 |

Sau khi tải xong, tất cả PDF được tổng hợp lên Google Drive vào thư mục `Dataset/`.

**Các loại báo cáo trong tập dữ liệu:**

| Loại báo cáo | Số lượng | Tỷ lệ |
|--------------|----------|-------|
| Annual Report | 426 | 72.6% |
| Sustainable Development Report | 66 | 11.2% |
| Sustainability Report | 56 | 9.5% |
| ESG Report | 8 | 1.4% |
| Integrated Annual Report | 7 | 1.2% |
| Integrated Report | 5 | 0.9% |
| Sustainability Development Report | 3 | 0.5% |
| Khác (Report of Sustainable Development, Report On Corporate Governance, ...) | 16 | 2.7% |

Tên báo cáo tuân theo mẫu: `{Company Name} - {Year} - {Report Type}`
Ví dụ: `FPT Corp - 2024 - Annual Report`, `Vingroup JSC - 2025 - Annual Report`

Khoảng thời gian: các báo cáo từ năm **2010 đến 2025**.

### 1.2 Size and Format

#### Dữ liệu thô (PDF)

- **587 file PDF** gốc trong thư mục `Dataset/`
- Kích thước: đã được xóa khỏi project do dung lượng lớn (theo `.gitignore`)

#### Dữ liệu ảnh (DatasetIMG)

Sau khi cắt PDF thành ảnh, dữ liệu được chia nhỏ để xử lý song song:

| Thành viên | Số lượng PDF được phân công cắt ảnh |
|------------|--------------------------------------|
| **Hào** | 122 + 116 = 238 PDF đầu |
| **Thống** | 116 PDF tiếp theo |
| **Bảo** | 116 PDF kế tiếp |
| **Thư** | 117 PDF cuối |

Mỗi thành viên tự upload kết quả cắt ảnh lên thư mục `DatasetIMG/` trên Google Drive, sau đó hợp nhất về máy chủ.

**Thống kê DatasetIMG:**

| Metric | Value |
|--------|-------|
| Tổng số thư mục con | **587** |
| Tổng số file ảnh | **63,807** |
| Tổng dung lượng | **~111.62 GB** |
| Định dạng ảnh | **JPG** |
| Độ phân giải | ZOOM=4 (3300×4200 px, phụ thuộc PDF gốc) |
| Số thư mục rỗng | 0 |
| File không phải JPG | 0 |

**Cấu trúc thư mục:**
```
DatasetIMG/
├── 3-2 Investment and Construction JSC - 2017 - Annual Report/
│   ├── 3_2_investment_and_construction_jsc_2017_p001.jpg
│   ├── 3_2_investment_and_construction_jsc_2017_p002.jpg
│   └── ...
├── FPT Corp - 2024 - Annual Report/
│   ├── fpt_corp_2024_p001.jpg
│   └── ...
└── (587 thư mục)
```

**Quy tắc đặt tên ảnh:** `{company_snake_case}_{year}_p{page_number_3digits}.jpg`
- `company_snake_case`: tên công ty viết thường, gạch dưới thay khoảng trắng, loại bỏ ký tự đặc biệt
- `year`: năm báo cáo (4 chữ số)
- `page_number_3digits`: số trang đánh từ 001

#### Dữ liệu OCR (DatasetOCR) — Đã xử lý

| Metric | Value |
|--------|-------|
| Tổng số file JSON | 886 + 1 state file |
| Số thư mục đã xử lý xong | 6 |
| Tổng số dòng text trích xuất | 41,503 |
| Tổng số ký tự trích xuất | 1,228,736 |

**Mỗi file JSON có cấu trúc:**
```json
[
  {
    "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "text": "Scope 1 emissions: 12,441 tCO2e",
    "confidence": 0.9873
  },
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `list[list[float]]` | 4 góc bounding box [top-left, top-right, bottom-right, bottom-left] |
| `text` | `string` | Văn bản được OCR nhận dạng |
| `confidence` | `float` | Độ tin cậy (0–1) |

### 1.3 Features

Tập dữ liệu ảnh đầu vào (DatasetIMG) có các đặc trưng:

| Feature | Type | Description |
|---------|------|-------------|
| `folder_name` | `string` | Tên thư mục = tên báo cáo gốc |
| `company_name` | `string` | Tên công ty (trích từ tên ảnh) |
| `year` | `integer` | Năm báo cáo (2017–2025) |
| `report_type` | `string` | Loại báo cáo (Annual Report, ESG Report, ...) |
| `page_number` | `integer` | Số trang (1-based) |
| `image_path` | `string` | Đường dẫn tuyệt đối file ảnh |
| `image_width` | `integer` | Chiều rộng ảnh (~3300px) |
| `image_height` | `integer` | Chiều cao ảnh (~4200px) |

Tập dữ liệu OCR output (DatasetOCR) bổ sung:

| Feature | Type | Description |
|---------|------|-------------|
| `bbox` | `list[list[float]]` | Tọa độ bounding box của dòng chữ |
| `text` | `string` | Nội dung văn bản |
| `confidence` | `float` | Độ tin cậy OCR (0–1) |
| `line_count` | `integer` | Số dòng text phát hiện được trên 1 trang |
| `is_empty` | `boolean` | True nếu không phát hiện text nào trên trang |

---

## 2. Data Cleaning and Preprocessing

### 2.1 Pipeline tổng thể

```
[1] Thu thập PDF
    ↓
[2] Chuẩn hóa tên folder
    ↓
[3] Cắt PDF → JPG (ZOOM=4)
    ↓
[4] PaddleOCR → JSON (đang chạy)
    ↓
[5] LayoutLMv3 + Table Transformer + ClimateBERT + RAG (kế hoạch)
```

### 2.2 Bước 1: Thu thập PDF

**Quy trình:** Nhóm 4 người chia nhau tải file PDF từ nguồn (sustainabilityreports.com và các nguồn báo cáo doanh nghiệp Việt Nam) dựa trên số trang, sau đó tổng hợp lên Google Drive.

**Thách thức:** 
- Một số PDF có tên chứa hậu tố `(SustainabilityReports.com)` gây trùng lặp và khó xử lý tự động.
- Nhiều báo cáo có tên không đồng nhất về định dạng ngày tháng, loại báo cáo.

**Giải pháp:** Script `namefolder.py` tự động loại bỏ hậu tố `(SustainabilityReports.com)` khỏi tên khi tạo thư mục.

### 2.3 Bước 2: Tạo cấu trúc thư mục

**Script:** `namefolder.py`

Tạo 587 thư mục trong `DatasetIMG/` song song với tên file PDF gốc, chuẩn hóa tên bằng cách loại bỏ hậu tố không cần thiết.

**Thách thức:** Tên PDF chứa khoảng trắng, ký tự đặc biệt, dấu câu — không thể dùng trực tiếp làm tên file an toàn.

**Giải pháp:** Chỉ chuẩn hóa ở mức tạo thư mục giữ tên gốc, việc snake_case hóa chỉ áp dụng cho tên file ảnh (xử lý trong `cut.py`).

### 2.4 Bước 3: Cắt PDF → JPG

**Script:** `cut.py` (folder-first) và `cutimg.py` (PDF-first)

**Quy trình:**
- Duyệt lần lượt 587 thư mục, tìm PDF tương ứng
- Mở từng file PDF bằng PyMuPDF (`fitz`)
- Render từng trang với độ phân giải ZOOM=4
- Lưu JPG vào thư mục tương ứng với tên chuẩn hóa

**Thách thức và giải pháp:**

| Thách thức | Giải pháp |
|------------|-----------|
| Dung lượng 111 GB — không thể chuyển qua Git | `.gitignore` loại trừ `DatasetIMG/` |
| 4 thành viên cắt song song, cần hợp nhất | Mỗi người cắt xong upload lên Drive, sau đó gộp thư mục |
| Tên ảnh cần đồng nhất để xử lý downstream | Dùng `create_company_name()` chuẩn hóa tên công ty về snake_case |
| Một số PDF bị lỗi khi đọc | Xử lý exception từng file, không làm dừng toàn bộ pipeline |

**Kết quả:** 63,807 ảnh JPG từ 587 PDF, tất cả đã được đặt tên đồng bộ, không có thư mục rỗng.

### 2.5 Bước 4: PaddleOCR — Đang chạy

**Script:** `paddle_ocr_batch.py`

**Quy trình:**
- Init PaddleOCR với `use_angle_cls=True, lang='en'`
- Duyệt từng thư mục trong `DatasetIMG/`
- Với mỗi ảnh JPG: OCR → JSON ghi vào `DatasetOCR/` song song
- Cơ chế resume qua file `.ocr_state.json`

**Thách thức và giải pháp:**

| Thách thức | Giải pháp |
|------------|-----------|
| Thời gian xử lý lâu (~31 giờ CPU cho 63,807 ảnh) | Cơ chế resume: ghi state sau mỗi thư mục, có thể dừng/chạy tiếp |
| Lỗi không tương thích phiên bản PaddlePaddle | Downgrade về `paddlepaddle==2.6.2 + paddleocr==2.7.3` |
| NumPy ABI conflict trên Windows | Ghép `numpy<2` với opencv tương thích |
| Một số trang không có text (chỉ biểu đồ/hình ảnh) | Ghi nhận là mảng rỗng trong JSON, không bỏ qua |

**Kết quả hiện tại:** 886/63,807 ảnh đã xử lý xong (6 thư mục đầu tiên), 41,503 dòng text trích xuất, 34 file JSON rỗng (trang không text).

### 2.6 Cơ chế Resume

File `.ocr_state.json` đóng vai trò checkpoint:
```json
{
  "completed_folders": [...],
  "completed_images": [...]
}
```

- `completed_folders`: danh sách thư mục đã xử lý xong toàn bộ ảnh
- `completed_images`: danh sách tên file ảnh đã xử lý

Khi chạy lại, script kiểm tra và bỏ qua các mục đã hoàn thành. Điều này cho phép:
- Dừng an toàn bất cứ lúc nào
- Chạy tiếp mà không mất công
- Phân phối lại công việc nếu cần

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Phân bố số trang theo báo cáo

```
Phân bố số trang/ảnh mỗi báo cáo (63,807 ảnh / 587 báo cáo):

Trung bình:  108.7 trang
Trung vị:    97 trang
Q1:          71 trang
Q3:          135 trang
Nhỏ nhất:    6 trang  (TNG Investment and Trading JSC - 2024)
Lớn nhất:    495 trang (Bao Viet Holdings - 2024 - Integrated Report)
```

```
Biểu đồ phân bố:

  0-10   trang:  ██ 6   (1.0%)
  11-30  trang:  ██ 11  (1.9%)
  31-50  trang:  ██████ 33 (5.6%)
  51-100 trang:  █████████████████████████████████████████ 255 (43.4%)
  101-200 trang: ██████████████████████████████████████ 241 (41.1%)
  201-500 trang: ████████ 41 (7.0%)
  501+   trang:  0 (0.0%)
```

**Nhận xét:** 84.5% báo cáo có độ dài từ 51–200 trang. Đây là kích thước điển hình của báo cáo thường niên / phát triển bền vững của doanh nghiệp Việt Nam. Tuy nhiên, có một số báo cáo rất ngắn (6 trang — có thể là báo cáo tóm tắt) và một số rất dài (495 trang — báo cáo tích hợp của tập đoàn lớn).

### 3.2 Phân bố loại báo cáo

```
Annual Report:                   ████████████████████████████████████████████████████ 426 (72.6%)
Sustainable Development Report:  █████████ 66 (11.2%)
Sustainability Report:           ████████ 56 (9.5%)
ESG Report:                      █ 8 (1.4%)
Integrated Annual Report:        █ 7 (1.2%)
Integrated Report:               █ 5 (0.9%)
Khác:                            ██ 19 (3.2%)
```

**Nhận xét:** Annual Report chiếm áp đảo (72.6%), phản ánh thực tế rằng hầu hết doanh nghiệp Việt Nam công bố thông tin ESG trong báo cáo thường niên thay vì báo cáo ESG riêng. Điều này đặt ra thách thức cho việc trích xuất: dữ liệu ESG nằm rải rác trong các phần khác nhau của báo cảo thường niên, không tập trung như báo cáo ESG chuyên biệt.

### 3.3 Phân bố theo năm

(Dựa trên tên thư mục — cần xác nhận từ dữ liệu đầy đủ)

Các báo cáo trải dài từ **2017 đến 2025**, tập trung chủ yếu ở các năm gần đây khi yêu cầu công bố ESG trở nên phổ biến.

### 3.4 Thống kê OCR (trên 886 ảnh đã xử lý)

```
Phân bố số dòng text mỗi trang:

Trung bình:  46.8 dòng
Trung vị:    40 dòng
Q1:          17 dòng
Q3:          65 dòng
Lớn nhất:    212 dòng
Nhỏ nhất:    0 dòng
File rỗng:   34 / 886 (3.8%)
```

```
Biểu đồ phân bố dòng/trang (886 ảnh mẫu):

  0 dòng     (rỗng):   ████ 34  (3.8%)
  1-20 dòng:           ████████████████ 156 (17.6%)
  21-50 dòng:          ████████████████████████████████ 314 (35.4%)
  51-100 dòng:         █████████████████████████████ 267 (30.1%)
  101-150 dòng:        ████████ 81 (9.1%)
  151-212 dòng:        █████ 34 (3.8%)
```

**Nhận xét:**
- 3.8% trang không có text — đây là các trang chỉ chứa hình ảnh, biểu đồ, infographic (phổ biến trong báo cáo ESG)
- Phần lớn trang có 20–100 dòng text, tương đương mật độ chữ vừa phải
- Trang có >150 dòng thường là các bảng số liệu dày đặc
- OCR confidence trung bình rất cao (>0.95 cho hầu hết các dòng), cho thấy ảnh ZOOM=4 đủ chất lượng
- Một số lỗi nhỏ: chữ Việt có dấu bị sai (do dùng `lang='en'`), số bị nhầm ký tự ở font nhỏ

### 3.5 Metrics tổng thể

| Metric | Raw Images (DatasetIMG) | OCR Output (DatasetOCR) |
|--------|------------------------|-------------------------|
| Records | 63,807 images | 886 JSON files (đang xử lý) |
| Size | ~111.62 GB | ~30 MB (ước tính ~2 GB khi hoàn thành) |
| Format | JPG | JSON |
| Features/file | ~5 (tên, năm, trang, đường dẫn, kích thước) | Tối thiểu 3 (bbox, text, confidence) |
| Empty rate | 0% | ~3.8% (trang không text) |
| Processing status | ✅ Hoàn thành | 🔄 Đang chạy |

### 3.6 Phát hiện chính (Key Findings)

1. **Dữ liệu không đồng nhất về độ dài:** Báo cáo dao động từ 6–495 trang. Pipeline cần xử lý linh hoạt, không hard-code giả định về số trang.

2. **Tỷ lệ trang không text cần xử lý đặc biệt:** ~3.8% trang chỉ có hình ảnh/biểu đồ — cần dùng LayoutLMv3 để hiểu ngữ cảnh hình ảnh thay vì chỉ dựa vào text OCR.

3. **Chất lượng OCR tốt nhưng chưa hoàn hảo:** Cần post-processing để sửa lỗi font nhỏ, chữ Việt có dấu. Có thể dùng PaddleOCR với `lang='en'` + từ điển tùy chỉnh để cải thiện độ chính xác.

4. **Cấu trúc thư mục sẵn sàng cho downstream:** Dữ liệu đã được tổ chức theo từng báo cáo, mỗi báo cáo là một thư mục riêng — thuận tiện cho việc xử lý theo batch và truy xuất theo chiều dọc (theo báo cáo) hoặc chiều ngang (theo ảnh).

5. **Cơ chế resume hoạt động hiệu quả:** Cho phép pipeline OCR chạy gián đoạn trên máy local mà không mất tiến độ — quan trọng khi xử lý 63,807 ảnh trên CPU (~31 giờ).
