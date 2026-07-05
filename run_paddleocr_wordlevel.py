r"""
run_paddleocr_wordlevel.py
Chay PaddleOCR tren dung tap anh da label vung (theo _annotations.coco.json),
lay ra text + bbox cho TUNG TU (khong phai tung vung), de sau nay merge voi
annotation vung tao label cho LayoutLMv3 token classification.

Co checkpoint/resume (giong pattern .ocr_state.json ban dang dung cho pipeline
OCR chinh), phong truong hop bi ngat giua chung tren 1921 anh.

Cai dat:
    pip install paddleocr paddlepaddle-gpu  --break-system-packages
    (dung paddlepaddle-gpu neu co GPU, hoac paddlepaddle ban CPU neu khong co)

Cach dung:
    python run_paddleocr_wordlevel.py --root "D:\Download\label.coco\train" \
        --out_json "D:\Download\label.coco\train\ocr_words.json"
"""

import argparse
import json
import os

from paddleocr import PaddleOCR


def quad_to_bbox(quad_pts):
    """PaddleOCR tra ve 4 diem goc (quadrilateral). Chuyen ve axis-aligned bbox [x0,y0,x1,y1]."""
    xs = [p[0] for p in quad_pts]
    ys = [p[1] for p in quad_pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state_path, state):
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp_path, state_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thu muc chua anh + _annotations.coco.json")
    ap.add_argument("--json_name", default="_annotations.coco.json")
    ap.add_argument("--out_json", required=True, help="File json ket qua OCR word-level")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    coco_path = os.path.join(args.root, args.json_name)
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    file_names = [im["file_name"] for im in coco["images"]]

    state_path = args.out_json + ".ocr_state.json"
    results = load_state(state_path)  # {file_name: [{"text":..,"bbox":[..]}]}

    remaining = [fn for fn in file_names if fn not in results]
    print(f"Tong so anh: {len(file_names)}, da xong: {len(results)}, con lai: {len(remaining)}")

    if remaining:
        ocr = PaddleOCR(
            use_gpu=False,
            lang="en",
            use_angle_cls=False,
            rec_batch_num=32,
            show_log=False
        )

        for i, fn in enumerate(remaining, 1):
            img_path = os.path.join(args.root, fn)
            if not os.path.exists(img_path):
                print(f"  [BO QUA] khong tim thay file: {fn}")
                results[fn] = []
                continue

            ocr_result = ocr.ocr(img_path, cls=False)
            words = []
            # ocr_result la list (1 phan tu cho 1 anh) cua list [box_pts, (text, conf)]
            page = ocr_result[0] if ocr_result and ocr_result[0] is not None else []
            for line in page:
                quad_pts, (text, conf) = line
                bbox = quad_to_bbox(quad_pts)
                words.append({"text": text, "bbox": bbox, "conf": float(conf)})
            results[fn] = words

            if i % 20 == 0 or i == len(remaining):
                save_state(state_path, results)
                print(f"  Da xu ly {i}/{len(remaining)} anh (checkpoint da luu)")

    save_state(state_path, results)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    n_empty = sum(1 for v in results.values() if len(v) == 0)
    print(f"\nHoan tat. Ket qua luu tai: {args.out_json}")
    print(f"So anh khong doc duoc chu nao (can kiem tra lai): {n_empty}")


if __name__ == "__main__":
    main()