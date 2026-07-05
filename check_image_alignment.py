r"""
check_image_alignment.py
Kiem tra xem anh da label tren Roboflow co khop kich thuoc (va ten file) voi
anh goc dang dua vao PaddleOCR hay khong. Neu lech, bbox cua 2 JSON se khong
cung he toa do va khong the merge truc tiep.

Cach dung:
    # Kiem tra co ban: kich thuoc trong json coco co khop voi file anh thuc te khong
    python check_image_alignment.py --roboflow_root "D:\Download\label.coco\train"

    # Neu co thu muc anh goc (nguon dua vao PaddleOCR) de doi chieu ten file + kich thuoc:
    python check_image_alignment.py --roboflow_root "D:\Download\label.coco\train" \
        --ocr_source_root "D:\ESG_Pages_Images"
"""

import argparse
import json
import os

from PIL import Image


def check_coco_vs_actual_files(root, json_name="_annotations.coco.json"):
    print("=== BUOC 1: Kich thuoc khai bao trong COCO json vs file anh thuc te ===")
    json_path = os.path.join(root, json_name)
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    mismatches = []
    missing = []
    for im in coco["images"]:
        fn = im["file_name"]
        path = os.path.join(root, fn)
        if not os.path.exists(path):
            missing.append(fn)
            continue
        with Image.open(path) as pil_img:
            actual_w, actual_h = pil_img.size
        if (actual_w, actual_h) != (im["width"], im["height"]):
            mismatches.append((fn, (im["width"], im["height"]), (actual_w, actual_h)))

    print(f"  Tong so anh trong json: {len(coco['images'])}")
    print(f"  File bi thieu tren dia: {len(missing)}")
    print(f"  Anh lech kich thuoc (json vs file thuc te): {len(mismatches)}")
    for fn, j, a in mismatches[:10]:
        print(f"    {fn}: json khai bao {j}, file thuc te {a}")
    return coco


def check_vs_ocr_source(coco, roboflow_root, ocr_source_root):
    print("\n=== BUOC 2: Doi chieu voi anh nguon OCR ===")
    n_found = 0
    n_missing = 0
    n_dim_mismatch = 0
    examples_mismatch = []
    examples_missing = []

    for im in coco["images"]:
        fn = im["file_name"]
        src_path = os.path.join(ocr_source_root, fn)
        if not os.path.exists(src_path):
            n_missing += 1
            if len(examples_missing) < 10:
                examples_missing.append(fn)
            continue
        n_found += 1
        with Image.open(src_path) as pil_img:
            src_w, src_h = pil_img.size
        if (src_w, src_h) != (im["width"], im["height"]):
            n_dim_mismatch += 1
            if len(examples_mismatch) < 10:
                examples_mismatch.append((fn, (im["width"], im["height"]), (src_w, src_h)))

    print(f"  Tim thay theo dung ten file trong OCR source: {n_found}/{len(coco['images'])}")
    print(f"  Khong tim thay (ten file khac nhau giua Roboflow va OCR source): {n_missing}")
    if examples_missing:
        print(f"  Vi du ten file khong khop: {examples_missing}")
    print(f"  Kich thuoc LECH giua Roboflow va OCR source: {n_dim_mismatch}")
    for fn, rf, src in examples_mismatch:
        print(f"    {fn}: Roboflow={rf}, OCR_source={src}")

    print("\n=== KET LUAN ===")
    if n_found == 0:
        print("  Khong khop duoc file nao theo ten -> ten file giua 2 nguon khac nhau.")
        print("  Can kiem tra lai quy uoc dat ten file (vi du Roboflow co the them hau to/hash vao ten).")
    elif n_dim_mismatch > 0:
        print(f"  {n_dim_mismatch} anh bi LECH kich thuoc -> Roboflow da resize/preprocess anh khac voi ban goc.")
        print("  Bat buoc phai OCR TREN CHINH anh da tai tu Roboflow (khong dung anh goc), hoac rescale lai bbox annotation theo ti le.")
    else:
        print(f"  Tat ca {n_found} anh khop ten + khop kich thuoc -> an toan de chay OCR truc tiep tren "
              f"anh trong thu muc OCR source va merge voi annotation Roboflow.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow_root", required=True, help="Thu muc chua anh + json tu Roboflow")
    ap.add_argument("--json_name", default="_annotations.coco.json")
    ap.add_argument("--ocr_source_root", default=None,
                     help="(Tuy chon) thu muc chua anh goc dung de OCR, de doi chieu")
    args = ap.parse_args()

    coco = check_coco_vs_actual_files(args.roboflow_root, args.json_name)

    if args.ocr_source_root:
        check_vs_ocr_source(coco, args.roboflow_root, args.ocr_source_root)
    else:
        print("\n(Khong truyen --ocr_source_root nen chua the doi chieu voi anh nguon OCR. "
              "Chay lai voi tham so nay tro toi thu muc anh goc de kiem tra day du.)")


if __name__ == "__main__":
    main()