"""
run_ocr_simple.py — PaddleOCR word-level extraction.

Usage:
    python run_ocr_simple.py --image-dir runs/{run_id}/pages --out-json runs/{run_id}/output/step1_ocr/ocr_words.json
"""
import argparse, gc, glob, json, os, sys, traceback
from paddleocr import PaddleOCR


def quad_to_bbox(quad_pts):
    xs = [p[0] for p in quad_pts]
    ys = [p[1] for p in quad_pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def main():
    ap = argparse.ArgumentParser(description="PaddleOCR word-level extraction")
    ap.add_argument("--image-dir", required=True, help="Directory containing page images")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--extensions", default=".jpg,.jpeg,.png", help="Image file extensions")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--gpu", action="store_true", help="Use GPU")
    ap.add_argument("--resume", action="store_true", help="Resume from previous state")
    args = ap.parse_args()

    exts = [e.strip().lower() for e in args.extensions.split(",")]
    image_files = sorted(
        p for p in glob.glob(os.path.join(args.image_dir, "*"))
        if os.path.splitext(p)[1].lower() in exts
    )
    image_names = sorted(os.path.basename(p) for p in image_files)

    print(f"Found {len(image_names)} images in {args.image_dir}")
    for fn in image_names:
        print(f"  - {fn}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    state_path = args.out_json + ".ocr_state.json"
    results = {}

    if args.resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} already done")

    remaining = [fn for fn in image_names if fn not in results]
    print(f"Remaining: {len(remaining)}")

    if remaining:
        ocr = None
        for i, fn in enumerate(remaining, 1):
            if ocr is None or i % 200 == 1:
                if ocr is not None:
                    del ocr
                    gc.collect()
                ocr = PaddleOCR(lang=args.lang, use_angle_cls=False,
                               rec_batch_num=8, use_gpu=args.gpu)

            img_path = os.path.join(args.image_dir, fn)
            if not os.path.exists(img_path):
                print(f"  Skipping {fn}: file not found")
                results[fn] = []
                continue

            try:
                ocr_result = ocr.ocr(img_path)
            except Exception as e:
                print(f"  Error OCR processing {fn}: {e}")
                traceback.print_exc()
                results[fn] = []
                continue

            words = []
            page = ocr_result[0] if ocr_result and ocr_result[0] is not None else []
            for line in page:
                quad_pts, (text, conf) = line
                bbox = quad_to_bbox(quad_pts)
                words.append({"text": text, "bbox": bbox, "conf": float(conf)})
            results[fn] = words

            if i % 20 == 0 or i == len(remaining):
                tmp = state_path + ".tmp"
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)
                os.replace(tmp, state_path)
                gc.collect()

    # Atomic write
    tmp_out = args.out_json + ".tmp"
    os.makedirs(os.path.dirname(tmp_out), exist_ok=True)
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp_out, args.out_json)

    n_empty = sum(1 for v in results.values() if len(v) == 0)
    print(f"Done. Output: {args.out_json}")
    print(f"Empty pages: {n_empty}")


if __name__ == "__main__":
    main()
