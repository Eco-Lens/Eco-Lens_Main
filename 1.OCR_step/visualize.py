"""
visualize_merge.py

Visualize OCR + merged labels after merge_ocr_with_regions.py

Ve:
- OCR word bbox
- text
- assigned label

Dung de QA dataset truoc khi train LayoutLMv3.

Cach dung:

python visualize_merge.py ^
    --root "valid" ^
    --merged_json "valid/layoutlmv3_dataset.json" ^
    --out_dir "valid/debug_vis" ^
    --max_images 50
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

# =========================
# COLOR MAP
# =========================

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
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )


def draw_text_background(draw, xy, text, color, font):
    bbox = draw.textbbox(xy, text, font=font)

    draw.rectangle(
        [
            bbox[0] - 2,
            bbox[1] - 2,
            bbox[2] + 2,
            bbox[3] + 2,
        ],
        fill=color,
    )

    draw.text(xy, text, fill=(0, 0, 0), font=font)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)
    ap.add_argument("--merged_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_images", type=int, default=50,
                     help="So anh toi da (0 = xoa het)")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.merged_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} merged samples")

    # random sample
    dataset = [x for x in dataset if len(x["words"]) > 10]
    print(f"Samples with OCR words: {len(dataset)}")

    # random sample (0 = het)
    if args.max_images > 0 and args.max_images < len(dataset):
        dataset = random.sample(dataset, args.max_images)

    # try load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for idx, sample in enumerate(dataset, 1):

        file_name = sample["file_name"]

        img_path = os.path.join(args.root, file_name)

        if not os.path.exists(img_path):
            print(f"[SKIP] missing image: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        draw = ImageDraw.Draw(image)

        words = sample["words"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        for word, box, label in zip(words, bboxes, labels):

            x0, y0, x1, y1 = box

            color = get_color(label)

            # bbox
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=color,
                width=2
            )

            # label
            draw_text_background(
                draw,
                (x0, max(0, y0 - 18)),
                label,
                color,
                font
            )

            # word
            draw_text_background(
                draw,
                (x0, y1 + 2),
                word,
                (255, 255, 255),
                font
            )

        out_path = os.path.join(
            args.out_dir,
            os.path.splitext(file_name)[0] + "_debug.jpg"
        )

        image.save(out_path)

        print(f"[{idx}] saved -> {out_path}")

    print("\nDone visualization.")


if __name__ == "__main__":
    main()
