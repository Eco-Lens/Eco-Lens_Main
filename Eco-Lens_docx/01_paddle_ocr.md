# Bước 1: PaddleOCR — Trích xuất văn bản từ ảnh báo cáo ESG

## 1. Tổng quan

Sau khi hoàn tất giai đoạn chuyển đổi 587 file PDF thành **63,807 ảnh JPG** (mỗi PDF có một thư mục riêng trong `DatasetIMG/`), bước tiếp theo là chạy **PaddleOCR** trên toàn bộ ảnh để trích xuất văn bản cùng tọa độ bounding box, phục vụ cho các bước xử lý downstream (LayoutLMv3, Table Transformer, ClimateBERT).

### Vai trò của PaddleOCR trong pipeline

```
PDF → [PaddleOCR] → Text + Bbox → LayoutLMv3 → Table Transformer → ClimateBERT → RAG → Output
         ↑
    Bước này
```

PaddleOCR đảm nhận việc chuyển đổi ảnh trang báo cáo thành văn bản máy đọc được, đồng thời giữ lại **thông tin vị trí (bounding box)** của từng dòng chữ — yếu tố sống còn để LayoutLMv3 có thể hiểu được bố cục tài liệu.

---

## 2. Lý do chọn PaddleOCR

Theo phân tích trong đề xuất dự án, PaddleOCR được chọn vì:

| Tiêu chí | Đánh giá |
|----------|----------|
| Đọc bảng biểu | Mạnh — hỗ trợ table structure tốt |
| Open-source | Có — dễ triển khai, không phụ thuộc cloud |
| Độ chính xác | Cao — kiến trúc Detection + Recognition Transformer |
| Đa ngôn ngữ | Tốt — hỗ trợ tiếng Việt và tiếng Anh |
| Scan PDF mờ | Ổn định — xử lý tốt tài liệu scan chất lượng thấp |
| Tích hợp pipeline | Dễ dàng — có thể nối tiếp với LayoutLMv3 |

So với các lựa chọn khác:
- **Tesseract OCR**: Quá yếu với complex layout của báo cáo ESG
- **EasyOCR**: Không tốt với bảng biểu
- **TrOCR**: Accuracy cao nhưng compute quá nặng
- **AWS Textract / Google Document AI**: Phụ thuộc cloud, chi phí lớn

---

## 3. Cài đặt PaddleOCR

### Yêu cầu hệ thống

- Python 3.8–3.11
- GPU NVIDIA với CUDA 11.x (khuyến nghị) hoặc chạy CPU
- RAM tối thiểu 16 GB (khuyến nghị 32 GB)
- Dung lượng ổ cứng trống tối thiểu 150 GB (cho dữ liệu + output)

### Cài đặt

```bash
# Tạo môi trường conda riêng
conda create -n ecolens python=3.10 -y
conda activate ecolens

# Cài paddlepaddle (chọn GPU hoặc CPU)
# GPU:
pip install paddlepaddle-gpu -U
# CPU:
pip install paddlepaddle -U

# Cài paddleocr
pip install paddleocr -U
```

### Kiểm tra cài đặt

```bash
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='en'); print('PaddleOCR ready')"
```

> **Lưu ý**: Mặc dù báo cáo ESG chủ yếu bằng tiếng Anh, bạn có thể cần `lang='en'` hoặc kết hợp thêm `lang='vi'` nếu gặp các báo cáo song ngữ. PaddleOCR hỗ trợ đa ngôn ngữ đồng thời.

---

## 4. Cấu trúc dữ liệu đầu vào

```
DatasetIMG/
├── 3-2 Investment and Construction JSC - 2017 - Annual Report/
│   ├── 3-2_investment_and_construction_jsc_2017_p001.jpg
│   ├── 3-2_investment_and_construction_jsc_2017_p002.jpg
│   └── ...
├── 577 Investment Corp - 2021 - Annual Report/
│   ├── 577_investment_corp_2021_p001.jpg
│   └── ...
├── FPT Corp - 2024 - Annual Report/
│   ├── fpt_corp_2024_p001.jpg
│   ├── fpt_corp_2024_p002.jpg
│   └── ...
├── Vingroup JSC - 2025 - Annual Report/
│   └── ...
└── ... (587 folders, 63,807 images total)
```

Đặc điểm:
- 587 thư mục — mỗi thư mục tương ứng một báo cáo PDF
- 63,807 file JPG — mỗi file là một trang báo cáo
- Độ phân giải: 4x (ZOOM=4, ~3300x4200px tùy theo PDF gốc)
- Tên file: `{company}_{year}_p{page:03d}.jpg`

---

## 5. Code xử lý OCR hàng loạt

### Script chính: `paddle_ocr_batch.py`

Tạo file `Eco-Lens_Main/paddle_ocr_batch.py`:

```python
import json
import time
from pathlib import Path

from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parent
IMG_DIR = PROJECT_ROOT / "DatasetIMG"
OCR_DIR = PROJECT_ROOT / "DatasetOCR"


def init_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=True,
        show_log=False,
        # Tăng độ chính xác cho tài liệu scan
        det_db_thresh=0.3,
        rec_batch_num=6,
    )


def process_image(ocr, image_path: Path) -> list:
    result = ocr.ocr(str(image_path), cls=True)
    if result is None or result[0] is None:
        return []
    return result[0]


def format_ocr_output(raw_boxes: list) -> list:
    lines = []
    for box_info in raw_boxes:
        bbox, (text, score) = box_info
        lines.append({
            "bbox": [[float(x), float(y)] for x, y in bbox],
            "text": text.strip(),
            "confidence": round(score, 4),
        })
    lines.sort(key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))
    return lines


def save_ocr_result(folder_name: str, image_stem: str, lines: list, ocr_dir: Path):
    out_dir = ocr_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    return out_path


def process_all_reports():
    ocr = init_ocr()
    start_time = time.time()

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    report_folders = sorted(
        [f for f in IMG_DIR.iterdir() if f.is_dir()]
    )
    total_images = 0
    total_errors = 0
    total_skipped = 0

    print(f"Found {len(report_folders)} report folders")
    print(f"Output dir: {OCR_DIR}")
    print("=" * 60)

    for idx, report_folder in enumerate(report_folders, start=1):
        folder_name = report_folder.name
        images = sorted(report_folder.glob("*.jpg"))
        if not images:
            print(f"[SKIP] {folder_name} — no images")
            total_skipped += 1
            continue

        print(f"\n[{idx}/{len(report_folders)}] {folder_name}")
        print(f"  Images: {len(images)}")

        for img_path in images:
            try:
                raw = process_image(ocr, img_path)
                lines = format_ocr_output(raw)
                save_ocr_result(folder_name, img_path.stem, lines, OCR_DIR)
                total_images += 1

                # Log mỗi 100 ảnh
                if len(lines) == 0:
                    print(f"  ⚠  {img_path.name}: no text detected")
                elif total_images % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  → {total_images} images done ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ✗ ERROR {img_path.name}: {e}")
                total_errors += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Done! {elapsed:.1f}s total")
    print(f"  Folders processed: {len(report_folders)}")
    print(f"  Images processed:  {total_images}")
    print(f"  Errors:            {total_errors}")
    print(f"  Skipped (no img):  {total_skipped}")
    print(f"  Output:            {OCR_DIR}")


if __name__ == "__main__":
    process_all_reports()
```

### Script ưu tiên GPU và checkpoint: `paddle_ocr_resume.py`

Để tránh mất công khi tiến trình bị gián đoạn (có thể mất nhiều ngày), dùng script có cơ chế resume:

```python
import json
import time
from pathlib import Path

from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parent
IMG_DIR = PROJECT_ROOT / "DatasetIMG"
OCR_DIR = PROJECT_ROOT / "DatasetOCR"
STATE_FILE = OCR_DIR / ".ocr_state.json"


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed_folders": [], "completed_images": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def init_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=True,
        show_log=False,
        det_db_thresh=0.3,
        rec_batch_num=6,
    )


def process_image(ocr, image_path: Path) -> list:
    result = ocr.ocr(str(image_path), cls=True)
    if result is None or result[0] is None:
        return []
    return result[0]


def format_ocr_output(raw_boxes: list) -> list:
    lines = []
    for box_info in raw_boxes:
        bbox, (text, score) = box_info
        lines.append({
            "bbox": [[float(x), float(y)] for x, y in bbox],
            "text": text.strip(),
            "confidence": round(score, 4),
        })
    lines.sort(key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))
    return lines


def save_ocr_result(folder_name: str, image_stem: str, lines: list, ocr_dir: Path):
    out_dir = ocr_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    return out_path


def process_all_reports():
    state = load_state()
    ocr = init_ocr()
    start_time = time.time()

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    completed_folders = set(state["completed_folders"])
    completed_images = set(state["completed_images"])

    report_folders = sorted([f for f in IMG_DIR.iterdir() if f.is_dir()])
    total_new = 0
    total_errors = 0
    total_skipped = 0

    print(f"Found {len(report_folders)} report folders")
    print(f"Already completed: {len(completed_folders)} folders")
    print(f"Output dir: {OCR_DIR}")
    print("=" * 60)

    for idx, report_folder in enumerate(report_folders, start=1):
        folder_name = report_folder.name

        images = sorted(report_folder.glob("*.jpg"))
        if not images:
            continue

        # Bỏ qua nếu folder đã hoàn thành
        if folder_name in completed_folders:
            print(f"[SKIP] {folder_name} — already done")
            total_skipped += 1
            continue

        print(f"\n[{idx}/{len(report_folders)}] {folder_name}")
        print(f"  Images: {len(images)}")

        for img_path in images:
            if img_path.name in completed_images:
                continue
            try:
                raw = process_image(ocr, img_path)
                lines = format_ocr_output(raw)
                save_ocr_result(folder_name, img_path.stem, lines, OCR_DIR)
                total_new += 1
                completed_images.add(img_path.name)

                if total_new % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  → {total_new} new images ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ✗ ERROR {img_path.name}: {e}")
                total_errors += 1

        completed_folders.add(folder_name)
        state = {
            "completed_folders": sorted(completed_folders),
            "completed_images": list(completed_images),
        }
        save_state(state)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Done! {elapsed:.1f}s total")
    print(f"  New images:        {total_new}")
    print(f"  Errors:            {total_errors}")
    print(f"  Completed folders: {len(completed_folders)}")
    print(f"  Output:            {OCR_DIR}")


if __name__ == "__main__":
    process_all_reports()
```

---

## 6. Cấu trúc dữ liệu đầu ra

Sau khi chạy xong, thư mục `DatasetOCR/` sẽ có cấu trúc song song với `DatasetIMG/`:

```
DatasetOCR/
├── 3-2 Investment and Construction JSC - 2017 - Annual Report/
│   ├── 3-2_investment_and_construction_jsc_2017_p001.json
│   ├── 3-2_investment_and_construction_jsc_2017_p002.json
│   └── ...
├── FPT Corp - 2024 - Annual Report/
│   ├── fpt_corp_2024_p001.json
│   ├── fpt_corp_2024_p002.json
│   └── ...
└── ...
```

Mỗi file JSON chứa danh sách các dòng văn bản được phát hiện, với cấu trúc:

```json
[
  {
    "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "text": "Scope 1 emissions: 12,441 tCO2e",
    "confidence": 0.98
  },
  {
    "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "text": "Scope 2 emissions: 5,221 tCO2e",
    "confidence": 0.96
  }
]
```

Trong đó:
- `bbox`: 4 góc của bounding box (góc trên-trái, trên-phải, dưới-phải, dưới-trái), định dạng [x, y]
- `text`: văn bản được nhận dạng
- `confidence`: độ tin cậy (0–1) của kết quả nhận dạng

Các dòng được sắp xếp theo thứ tự đọc từ trên xuống dưới, từ trái sang phải.

---

## 7. Ước tính thời gian xử lý

| Cấu hình | Thời gian ước tính cho 63,807 ảnh |
|----------|----------------------------------|
| GPU RTX 3060 + CUDA | ~8–15 giờ |
| GPU RTX 4090 + CUDA | ~4–8 giờ |
| CPU (32 cores) | ~3–7 ngày |

> **Khuyến nghị**: Sử dụng GPU để giảm thời gian xử lý xuống còn 1–2 ca làm việc. CPU sẽ mất vài ngày và dễ bị gián đoạn.

---

## 8. Xử lý sự cố thường gặp

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| OCR ra text rỗng | Ảnh quá mờ hoặc font quá nhỏ | Kiểm tra ảnh, tăng det_db_thresh, thử `det_db_thresh=0.1` |
| Bbox bị thiếu dòng | Scan PDF bị lệch | Dùng `use_angle_cls=True` để tự động xoay ảnh |
| Memory full | Batch quá lớn | Giảm `rec_batch_num` xuống 4 hoặc 2 |
| Tiếng Việt sai dấu | Lang không phù hợp | Thêm `lang='vi'` hoặc `lang='en'` tùy ngôn ngữ báo cáo |
| Tiến trình treo | GPU OOM | Giảm batch, set `PADDLE_DEVICE=0` cho 1 GPU |

---

## 9. Kiểm tra chất lượng OCR

Sau khi hoàn thành, chạy script kiểm tra nhanh:

```python
from pathlib import Path
import json

OCR_DIR = Path("DatasetOCR")
total_lines = 0
empty_files = 0
total_files = 0

for json_file in OCR_DIR.rglob("*.json"):
    total_files += 1
    with open(json_file) as f:
        data = json.load(f)
    if not data:
        empty_files += 1
    total_lines += len(data)

print(f"Files:      {total_files}")
print(f"Empty:      {empty_files} ({empty_files/total_files*100:.1f}%)")
print(f"Total lines: {total_lines}")
```

Dự kiến:
- Tỷ lệ file rỗng < 3% (một số trang chỉ có hình ảnh/biểu đồ không có text)
- Trung bình 20–80 dòng/phát hiện mỗi trang tùy độ phức tạp của báo cáo

---

## 10. Tích hợp với bước tiếp theo (LayoutLMv3)

Sau khi có OCR output, dữ liệu được chuyển sang **Bước 2: LayoutLMv3**. LayoutLMv3 cần 3 đầu vào:

| Đầu vào | Nguồn | Mô tả |
|---------|-------|-------|
| **Text embedding** | PaddleOCR output (`text`) | Nội dung văn bản từng dòng |
| **Position embedding** | PaddleOCR output (`bbox`) | Tọa độ bounding box trên trang |
| **Image embedding** | Ảnh JPG gốc | Hình ảnh trực quan của trang |

Định dạng JSON hiện tại đã lưu đầy đủ `text` và `bbox`, sẵn sàng để LayoutLMv3 sử dụng trực tiếp.

---

## 11. Tổng kết

Sau bước này, bạn sẽ có:
- ✓ 63,807 file JSON (mỗi file tương ứng 1 trang báo cáo)
- ✓ Mỗi JSON chứa danh sách text + bounding box + confidence
- ✓ Cấu trúc thư mục giữ nguyên song song với `DatasetIMG/`
- ✓ Dữ liệu sẵn sàng cho LayoutLMv3 ở Bước 2

**Tổng dung lượng đầu ra dự kiến:** ~500 MB – 2 GB (tùy độ dài văn bản mỗi trang).
