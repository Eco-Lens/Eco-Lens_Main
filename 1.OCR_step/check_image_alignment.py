r"""
check_image_alignment.py
Kiem tra xem anh da label tren Roboflow co khop kich thuoc (va ten file) voi
anh goc dang dua vao PaddleOCR hay khong. Neu lech, bbox cua 2 JSON se khong
cung he toa do va khong the merge truc tiep.

Khi so sanh voi thu muc anh goc (ocr_source_root), script tu dong:
  - Duyet de quy cac thu muc con (vi du DatasetIMG_Reduce chua cac thu muc
    theo tung cong ty).
  - Ghep ten file bang truong extra.name trong COCO JSON thay vi dung
    file_name (vi Roboflow them hash vao ten file).

Cach dung:
    # Kiem tra co ban: kich thuoc trong json coco co khop voi file anh thuc te khong
    python check_image_alignment.py --roboflow_root "valid"

    # Neu co thu muc anh goc (nguon dua vao PaddleOCR) de doi chieu ten file + kich thuoc:
    python check_image_alignment.py --roboflow_root "valid" \
        --ocr_source_root "DatasetIMG_Reduce"
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


def _build_ocr_file_map(ocr_source_root):
    """Duyet de quy thu muc ocr_source_root, tra ve dict {original_name: path}."""
    name_to_path = {}
    for dirpath, _, filenames in os.walk(ocr_source_root):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                name_to_path[fn] = os.path.join(dirpath, fn)
    return name_to_path


def check_vs_ocr_source(coco, roboflow_root, ocr_source_root):
    print("\n=== BUOC 2: Doi chieu voi anh nguon OCR ===")
    ocr_file_map = _build_ocr_file_map(ocr_source_root)
    print(f"  Tim thay {len(ocr_file_map)} file anh trong thu muc OCR source (de quy)")

    n_found = 0
    n_missing = 0
    n_dim_mismatch = 0
    examples_missing = []
    examples_mismatch = []

    for im in coco["images"]:
        extra_name = im.get("extra", {}).get("name", None)
        if extra_name is None:
            fn = im["file_name"]
        else:
            fn = extra_name
        src_path = ocr_file_map.get(fn)
        if src_path is None:
            n_missing += 1
            if len(examples_missing) < 10:
                examples_missing.append((im["file_name"], fn))
            continue
        n_found += 1
        with Image.open(src_path) as pil_img:
            src_w, src_h = pil_img.size
        if (src_w, src_h) != (im["width"], im["height"]):
            n_dim_mismatch += 1
            if len(examples_mismatch) < 10:
                examples_mismatch.append((fn, (im["width"], im["height"]), (src_w, src_h)))

    print(f"  File tim thay (ghep qua extra.name hoac file_name): {n_found}/{len(coco['images'])}")
    print(f"  File KHONG tim thay trong OCR source: {n_missing}")
    if examples_missing:
        print(f"  Vi du file khong tim thay (Roboflow name -> original name):")
        for rf_name, orig_name in examples_missing:
            print(f"    {rf_name}  ->  {orig_name}")
    print(f"  Kich thuoc LECH giua Roboflow va OCR source: {n_dim_mismatch}")
    for fn, rf, src in examples_mismatch:
        print(f"    {fn}: Roboflow={rf}, OCR_source={src}")

    print("\n=== KET LUAN ===")
    if n_found == 0:
        print("  Khong tim thay file nao trong thu muc OCR source.")
        print("  Kiem tra extra.name trong COCO JSON co dung khong, hoac OCR source co chua dung anh khong.")
    elif n_dim_mismatch > 0:
        print(f"  {n_dim_mismatch} anh bi LECH kich thuoc -> Roboflow da resize/preprocess anh khac voi anh goc.")
        print("  Bat buoc phai OCR TREN CHINH anh tu Roboflow (trong valid/), hoac rescale lai annotation.")
    else:
        print(f"  Tat ca {n_found} anh khop ten + khop kich thuoc -> co the dung anh goc de OCR.")


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