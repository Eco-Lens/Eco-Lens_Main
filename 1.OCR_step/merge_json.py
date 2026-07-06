r"""
merge_ocr_with_regions.py
Merge OCR word-level (tu run_paddleocr_wordlevel.py) voi annotation vung
(0_annotations.coco.json tu Roboflow) de tao du lieu token classification
cho LayoutLMv3: moi tu (word) duoc gan 1 label = ten vung ma no nam trong.

Quy tac gan nhan:
  - Dung DIEM TAM (center point) cua tu de kiem tra nam trong vung nao,
    on dinh hon so voi kiem tra overlap toan phan o gan bien.
  - Neu tu khong roi vao vung annotation nao -> gan nhan mac dinh "O".
  - Neu tu roi vao nhieu vung chong lan nhau -> uu tien vung co DIEN TICH
    NHO HON (vung cu the hon, vi du "table_text" long trong "table" lon).

Cach dung:
    python merge_ocr_with_regions.py \
        --root "valid" \
        --ocr_json "valid/0_ocr_words.json" \
        --out_json "valid/0_layoutlmv3_dataset.json"
"""
import argparse
import json
import os


def point_in_box(px, py, box):
    x0, y0, x1, y1 = box
    return x0 <= px <= x1 and y0 <= py <= y1


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def assign_label(word_bbox, region_boxes, default_label="O"):
    """region_boxes: list cua (label_name, [x0,y0,x1,y1]).
    Tra ve label cua vung nho nhat ma tam cua tu roi vao."""
    wx0, wy0, wx1, wy1 = word_bbox
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2

    candidates = []
    for label_name, box in region_boxes:
        if point_in_box(cx, cy, box):
            candidates.append((box_area(box), label_name))

    if not candidates:
        return default_label
    candidates.sort(key=lambda x: x[0])  # dien tich nho nhat truoc
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json_name", default="0_annotations.coco.json")
    ap.add_argument("--ocr_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--default_label", default="O")
    args = ap.parse_args()

    coco_path = os.path.join(args.root, args.json_name)
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(args.ocr_json, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im for im in coco["images"]}

    regions_by_img = {}  # image_id -> list of (label_name, [x0,y0,x1,y1])
    for a in coco["annotations"]:
        box_xyxy = coco_bbox_to_xyxy(a["bbox"])
        label_name = cat_by_id[a["category_id"]]
        regions_by_img.setdefault(a["image_id"], []).append((label_name, box_xyxy))

    dataset = []
    n_words_total = 0
    n_words_default = 0
    label_counter = {}

    for img_id, img_info in img_by_id.items():
        fn = img_info["file_name"]
        words_info = ocr_data.get(fn, [])
        region_boxes = regions_by_img.get(img_id, [])

        words, bboxes, labels = [], [], []
        for w in words_info:
            label = assign_label(w["bbox"], region_boxes, args.default_label)
            words.append(w["text"])
            bboxes.append(w["bbox"])
            labels.append(label)
            label_counter[label] = label_counter.get(label, 0) + 1
            n_words_total += 1
            if label == args.default_label:
                n_words_default += 1

        dataset.append({
            "image_id": img_id,
            "file_name": fn,
            "width": img_info["width"],
            "height": img_info["height"],
            "words": words,
            "bboxes": bboxes,
            "labels": labels,
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    print(f"Da tao {len(dataset)} anh, tong {n_words_total} tu.")
    print(f"So tu roi vao nhan mac dinh '{args.default_label}' (khong thuoc vung nao): "
          f"{n_words_default} ({100*n_words_default/max(n_words_total,1):.1f}%)")
    print("\nPhan bo label sau merge:")
    for k, v in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}")
    print(f"\nKet qua luu tai: {args.out_json}")


if __name__ == "__main__":
    main()