import json
import time
from pathlib import Path

from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = PROJECT_ROOT / "DatasetIMG"
OCR_DIR = PROJECT_ROOT / "DatasetOCR"
STATE_FILE = OCR_DIR / ".ocr_state.json"


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_folders": [], "completed_images": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def init_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
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


def save_ocr_result(folder_name: str, image_stem: str, lines: list):
    out_dir = OCR_DIR / folder_name
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

    completed_folders = set(state.get("completed_folders", []))
    completed_images = set(state.get("completed_images", []))

    report_folders = sorted([f for f in IMG_DIR.iterdir() if f.is_dir()])
    total_new = 0
    total_errors = 0
    total_skipped = 0

    print(f"Found {len(report_folders)} report folders")
    print(f"Already completed: {len(completed_folders)} folders, {len(completed_images)} images")
    print(f"Output dir: {OCR_DIR}")
    print("=" * 60)

    for idx, report_folder in enumerate(report_folders, start=1):
        folder_name = report_folder.name
        images = sorted(report_folder.glob("*.jpg"))
        if not images:
            continue

        if folder_name in completed_folders:
            print(f"[SKIP] ({idx}/{len(report_folders)}) {folder_name} — already done")
            total_skipped += 1
            continue

        print(f"\n[{idx}/{len(report_folders)}] {folder_name} ({len(images)} images)")

        for img_path in images:
            if img_path.name in completed_images:
                continue
            try:
                raw = process_image(ocr, img_path)
                lines = format_ocr_output(raw)
                save_ocr_result(folder_name, img_path.stem, lines)
                total_new += 1
                completed_images.add(img_path.name)
            except Exception as e:
                print(f"  ERROR {img_path.name}: {e}")
                total_errors += 1

        completed_folders.add(folder_name)
        state = {
            "completed_folders": sorted(completed_folders),
            "completed_images": list(completed_images),
        }
        save_state(state)

        elapsed = time.time() - start_time
        print(f"  -> Done. Total new: {total_new}, Errors: {total_errors}, Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"FINISHED! {elapsed:.1f}s total")
    print(f"  New images:        {total_new}")
    print(f"  Errors:            {total_errors}")
    print(f"  Skipped folders:   {total_skipped}")
    print(f"  Completed folders: {len(completed_folders)}")
    print(f"  Output:            {OCR_DIR}")


if __name__ == "__main__":
    process_all_reports()
