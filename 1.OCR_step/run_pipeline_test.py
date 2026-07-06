"""
run_pipeline_test.py
Chay thu pipeline OCR + merge + visualize tren 1 tap nho
de kiem tra chat luong truoc khi chay toan bo dataset.

Pipeline:
  1. Lay N anh ngau nhien tu COCO JSON
  2. PaddleOCR word-level
  3. Merge OCR voi region annotation
  4. Visualize debug

Cach dung:
    python 1.OCR_step/run_pipeline_test.py --root "valid" --max_images 10
"""

import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

# them thu muc hien tai vao path de co the import neu can
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_paddleocr_wordlevel import quad_to_bbox
from merge_json import point_in_box, box_area, coco_bbox_to_xyxy

LABEL_COLORS = {
    "O": (160, 160, 160),
    "header": (255, 0, 0),
    "title": (255, 80, 80),
    "body_content": (255, 200, 0),
    "table": (0, 180, 255),
    "table_text": (0, 120, 255),
    "footer": (255, 0, 255),
    "figure": (0, 255, 0),
    "caption": (0, 200, 0),
}


def get_color(label):
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    random.seed(hash(label) % 100000)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_text_background(draw, xy, text, color, font):
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def assign_label(word_bbox, region_boxes, default_label="O"):
    wx0, wy0, wx1, wy1 = word_bbox
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    candidates = []
    for label_name, box in region_boxes:
        if point_in_box(cx, cy, box):
            candidates.append((box_area(box), label_name))
    if not candidates:
        return default_label
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser(description="Test OCR pipeline on a small sample")
    ap.add_argument("--root", default="valid", help="Thu muc chua anh + _annotations.coco.json")
    ap.add_argument("--json_name", default="_annotations.coco.json")
    ap.add_argument("--max_images", type=int, default=10, help="So anh thu (mac dinh 10)")
    ap.add_argument("--out_prefix", default="test", help="Tien to cho file output")
    ap.add_argument("--seed", type=int, default=42, help="Random seed de tai lap")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    random.seed(args.seed)
    root = args.root
    out_ocr = os.path.join(root, f"{args.out_prefix}_ocr_words.json")
    out_merged = os.path.join(root, f"{args.out_prefix}_layoutlmv3_dataset.json")
    out_vis = os.path.join(root, f"{args.out_prefix}_debug_vis")

    # --- DOC COCO JSON ---
    coco_path = os.path.join(root, args.json_name)
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    images_all = coco["images"]
    annotations_all = coco["annotations"]

    # --- CHON NGAY NHIEN N ANH ---
    n = min(args.max_images, len(images_all))
    selected = random.sample(images_all, n)
    selected_ids = {im["id"] for im in selected}
    print(f"Chon {n} anh ngau nhien tu {len(images_all)} anh tong cong")
    for im in selected:
        print(f"  - id={im['id']}: {im['file_name']}")

    # --- BUOC 1: OCR ---
    print(f"\n=== BUOC 1: OCR word-level ({n} anh) ===")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_gpu=False, lang=args.lang,
                        use_angle_cls=False, rec_batch_num=32, show_log=False)
    except ImportError as e:
        print(f"LOI: Khong the import PaddleOCR. Da cai dat chua?")
        print(f"  pip install paddleocr")
        sys.exit(1)

    ocr_results = {}
    for i, im in enumerate(selected, 1):
        fn = im["file_name"]
        img_path = os.path.join(root, fn)
        if not os.path.exists(img_path):
            print(f"  [{i}/{n}] BO QUA: {fn} (khong tim thay file)")
            ocr_results[fn] = []
            continue

        result = ocr.ocr(img_path, cls=False)
        words = []
        page = result[0] if result and result[0] is not None else []
        for line in page:
            quad_pts, (text, conf) = line
            bbox = quad_to_bbox(quad_pts)
            words.append({"text": text, "bbox": bbox, "conf": float(conf)})
        ocr_results[fn] = words
        print(f"  [{i}/{n}] {fn}: {len(words)} tu")

    with open(out_ocr, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False)
    print(f"  OCR ket qua luu tai: {out_ocr}")

    # --- BUOC 2: MERGE ---
    print(f"\n=== BUOC 2: Merge OCR voi region annotation ===")
    regions_by_img = {}
    for a in annotations_all:
        if a["image_id"] in selected_ids:
            box_xyxy = coco_bbox_to_xyxy(a["bbox"])
            label_name = cat_by_id[a["category_id"]]
            regions_by_img.setdefault(a["image_id"], []).append((label_name, box_xyxy))

    dataset = []
    default_label = "O"
    label_counter = {}
    n_words_total = 0
    n_words_default = 0

    for im in selected:
        fn = im["file_name"]
        words_info = ocr_results.get(fn, [])
        region_boxes = regions_by_img.get(im["id"], [])

        words, bboxes, labels = [], [], []
        for w in words_info:
            label = assign_label(w["bbox"], region_boxes, default_label)
            words.append(w["text"])
            bboxes.append(w["bbox"])
            labels.append(label)
            label_counter[label] = label_counter.get(label, 0) + 1
            n_words_total += 1
            if label == default_label:
                n_words_default += 1

        dataset.append({
            "image_id": im["id"],
            "file_name": fn,
            "width": im["width"],
            "height": im["height"],
            "words": words,
            "bboxes": bboxes,
            "labels": labels,
        })

    with open(out_merged, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    print(f"Da merge {len(dataset)} anh, tong {n_words_total} tu.")
    print(f"So tu 'O' (khong thuoc region nao): {n_words_default} "
          f"({100 * n_words_default / max(n_words_total, 1):.1f}%)")
    print("Phan bo label:")
    for k, v in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}")
    print(f"Merge ket qua luu tai: {out_merged}")

    # --- BUOC 3: VISUALIZE ---
    print(f"\n=== BUOC 3: Visualize debug ===")
    os.makedirs(out_vis, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, sample in enumerate(dataset, 1):
        fn = sample["file_name"]
        img_path = os.path.join(root, fn)
        if not os.path.exists(img_path):
            print(f"  [{idx}] SKIP: {fn}")
            continue

        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for word, box, label in zip(sample["words"], sample["bboxes"], sample["labels"]):
            x0, y0, x1, y1 = box
            color = get_color(label)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw_text_background(draw, (x0, max(0, y0 - 18)), label, color, font)
            draw_text_background(draw, (x0, y1 + 2), word, (255, 255, 255), font)

        out_path = os.path.join(out_vis, os.path.splitext(fn)[0] + "_debug.jpg")
        image.save(out_path)
        print(f"  [{idx}/{len(dataset)}] {out_path}")

    print(f"\n=== HOAN TAT ===")
    print(f"  OCR JSON:    {out_ocr}")
    print(f"  Merged JSON: {out_merged}")
    print(f"  Debug images:{out_vis}/")
    print(f"\nKiem tra debug images truoc khi chay pipeline full.")
    print(f"De chay full dataset, dung cac lenh rieng:")
    print(f"  python 1.OCR_step/run_paddleocr_wordlevel.py --root \"{root}\" --out_json \"{root}/ocr_words.json\"")
    print(f"  python 1.OCR_step/merge_json.py --root \"{root}\" --ocr_json \"{root}/ocr_words.json\" --out_json \"{root}/layoutlmv3_dataset.json\"")
    print(f"  python 1.OCR_step/visualize.py --root \"{root}\" --merged_json \"{root}/layoutlmv3_dataset.json\" --out_dir \"{root}/debug_vis\"")


if __name__ == "__main__":
    main()
