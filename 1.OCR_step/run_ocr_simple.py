"""
run_ocr_simple.py
PaddleOCR word-level truc tiep tren thu muc anh, khong can file annotation.

Cach dung:
    python "1.OCR_step/run_ocr_simple.py" --root "test" --out_json "test/0_ocr_words.json"
"""
import argparse
import gc
import glob
import json
import os
from paddleocr import PaddleOCR


def quad_to_bbox(quad_pts):
    xs = [p[0] for p in quad_pts]
    ys = [p[1] for p in quad_pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def main():
    ap = argparse.ArgumentParser(description="PaddleOCR word-level (khong can COCO)")
    ap.add_argument("--root", required=True, help="Thu muc chua anh")
    ap.add_argument("--out_json", required=True, help="File output")
    ap.add_argument("--extensions", default=".jpg,.jpeg,.png,.tiff,.bmp", help="Duoi mo rong")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    exts = [e.strip().lower() for e in args.extensions.split(",")]
    image_paths = sorted(glob.glob(os.path.join(args.root, "*")))
    image_files = [p for p in image_paths if os.path.splitext(p)[1].lower() in exts]
    image_names = sorted(os.path.basename(p) for p in image_files)

    print(f"Tim thay {len(image_names)} anh trong {args.root}")
    for fn in image_names:
        print(f"  - {fn}")

    state_path = args.out_json + ".ocr_state.json"
    results = {}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    remaining = [fn for fn in image_names if fn not in results]
    print(f"\nDa xong: {len(results)}, con lai: {len(remaining)}")

    if remaining:
        ocr = None
        for i, fn in enumerate(remaining, 1):
            if ocr is None or i % 200 == 1:
                if ocr is not None:
                    del ocr
                    gc.collect()
                ocr = PaddleOCR( lang=args.lang,
                                use_angle_cls=False, rec_batch_num=8)

            img_path = os.path.join(args.root, fn)
            if not os.path.exists(img_path):
                results[fn] = []
                continue

            ocr_result = ocr.ocr(img_path)
            words = []
            page = ocr_result[0] if ocr_result and ocr_result[0] is not None else []
            for line in page:
                quad_pts, (text, conf) = line
                bbox = quad_to_bbox(quad_pts)
                words.append({"text": text, "bbox": bbox, "conf": float(conf)})
            results[fn] = words

            if i % 20 == 0 or i == len(remaining):
                tmp = state_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)
                os.replace(tmp, state_path)
                gc.collect()

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    n_empty = sum(1 for v in results.values() if len(v) == 0)
    print(f"\nHoan tat. Output: {args.out_json}")
    print(f"Anh khong co chu: {n_empty}")


if __name__ == "__main__":
    main()