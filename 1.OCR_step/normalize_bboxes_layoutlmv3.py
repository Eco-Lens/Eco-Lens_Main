"""
normalize_bboxes_layoutlmv3.py
Chuan hoa bbox (pixel) trong file dataset da merge (words/bboxes/labels/width/height)
ve thang 0-1000, dung format input chuan cua LayoutLMv3.

Cach dung:
python ".\1.OCR_step\normalize_bboxes_layoutlmv3.py" --in_json "output\OCR_Merge_LabelRegion_preLayoutmv3\0_layoutlmv3_dataset.json" --out_json "output\OCR_Merge_LabelRegion_preLayoutmv3\0_layoutlmv3_dataset_normalized.json"
"""
import argparse
import json
def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox

    # Clamp truoc khi normalize, phong truong hop bbox annotation
    # bi lech ra ngoai bien anh (~112 case da phat hien luc QC)
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))

    # Dam bao x1 >= x0, y1 >= y0 sau khi clamp
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    nx0 = round(1000 * x0 / width)
    ny0 = round(1000 * y0 / height)
    nx1 = round(1000 * x1 / width)
    ny1 = round(1000 * y1 / height)

    # Clamp lan cuoi de tuyet doi khong vuot [0,1000] (tranh loi lam tron)
    nx0, nx1 = max(0, min(nx0, 1000)), max(0, min(nx1, 1000))
    ny0, ny1 = max(0, min(ny0, 1000)), max(0, min(ny1, 1000))

    return [nx0, ny0, nx1, ny1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    n_clamped = 0
    n_boxes_total = 0

    for item in dataset:
        w, h = item["width"], item["height"]
        new_bboxes = []
        for bbox in item["bboxes"]:
            x0, y0, x1, y1 = bbox
            was_out_of_bounds = x0 < 0 or y0 < 0 or x1 > w or y1 > h
            norm = normalize_bbox(bbox, w, h)
            if was_out_of_bounds:
                n_clamped += 1
            new_bboxes.append(norm)
            n_boxes_total += 1
        item["bboxes"] = new_bboxes
        # Giu lai width/height goc de truy vet neu can, khong bat buoc cho LayoutLMv3
        item["bbox_scale"] = "0-1000"

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    print(f"Tong so bbox: {n_boxes_total}")
    print(f"So bbox bi clamp (vuot bien anh truoc khi normalize): {n_clamped} "
          f"({100*n_clamped/max(n_boxes_total,1):.2f}%)")
    print(f"Ket qua luu tai: {args.out_json}")


if __name__ == "__main__":
    main()