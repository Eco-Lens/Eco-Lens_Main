"""
inference_layoutlmv3.py
------------------------
Load model từ Output/2_Model_Layoutlmv3_Finetune/checkpoint-1000/
Inference trên OCR words của tập test/ -> gán label cho mỗi word.

Cách dùng:
    python "2.LayoutLMV3_step/inference_layoutlmv3.py"
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
import transformers.utils.generic as generic
generic.is_tf_available = lambda: False

from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification
from PIL import Image

CKPT = os.path.join("Output", "2_Model_Layoutlmv3_Finetune", "checkpoint-1000")
OCR_JSON = os.path.join("test", "0_ocr_words.json")
LABEL_OUT = os.path.join("test", "0_layoutlmv3_labels.json")
IMAGE_DIR = "test"

ID2LABEL = {
    0: "O", 1: "chart", 2: "figure", 3: "footer", 4: "header",
    5: "ignore", 6: "table", 7: "table_text", 8: "text", 9: "toc"
}

_img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def normalize_bbox(bbox, w, h):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return [round(1000 * x0 / w), round(1000 * y0 / h),
            round(1000 * x1 / w), round(1000 * y1 / h)]


def main():
    t0 = time.time()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = LayoutLMv3ForTokenClassification.from_pretrained(CKPT)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Model loaded on {device} in {time.time()-t0:.0f}s")

    with open(OCR_JSON, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    results = {}

    for img_name, words_list in sorted(ocr_data.items()):
        if not words_list:
            results[img_name] = []
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_name} not found")
            results[img_name] = []
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        texts = [w["text"] for w in words_list]
        bboxes_pixel = [w["bbox"] for w in words_list]
        bboxes_norm = [normalize_bbox(b, w, h) for b in bboxes_pixel]

        encoding = tokenizer(
            text=texts,
            boxes=bboxes_norm,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        )

        pixel_values = _img_transform(img).unsqueeze(0).to(device)
        encoding["pixel_values"] = pixel_values
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        predictions = outputs.logits.argmax(dim=-1).squeeze(0).tolist()
        token_word_ids = encoding["input_ids"].squeeze(0).tolist()

        word_labels = [0] * len(texts)
        for token_pos, wid in enumerate(token_word_ids):
            if wid is None or wid < 0 or wid >= len(texts):
                continue
            if predictions[token_pos] == -100:
                continue
            word_labels[wid] = predictions[token_pos]

        label_names = [ID2LABEL.get(l, "O") for l in word_labels]
        n_table = sum(1 for l in label_names if l in ("table", "table_text"))
        results[img_name] = label_names
        print(f"  {img_name}: {len(texts)} words, {n_table} table/table_text")

    with open(LABEL_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved labels to {LABEL_OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
