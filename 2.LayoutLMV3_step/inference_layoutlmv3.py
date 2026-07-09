"""
inference_layoutlmv3.py
------------------------
Load LayoutLMv3 model từ checkpoint fine-tuned.
Inference bằng LayoutLMv3Processor (image + words + boxes).
Chunk words thành từng nhóm 60 words (no overlap) để tránh max_length=512.

Cách dùng:
    python "2.LayoutLMV3_step/inference_layoutlmv3.py"
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import Counter
from PIL import Image

import transformers.utils.generic as generic
generic.is_tf_available = lambda: False

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

CKPT = os.path.join("Output", "2_Model_Layoutlmv3_Finetune", "checkpoint-1000")
OCR_JSON = os.path.join("test", "0_ocr_words.json")
LABEL_OUT = os.path.join("test", "0_layoutlmv3_labels.json")
IMAGE_DIR = "test"

CHUNK_SIZE = 60
MAX_LEN = 512

ID2LABEL = {
    0: "O", 1: "chart", 2: "figure", 3: "footer", 4: "header",
    5: "ignore", 6: "table", 7: "table_text", 8: "text", 9: "toc",
}


def clean_bbox(bbox, w, h):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    # Normalize to 0-1000
    return [round(1000 * x0 / w), round(1000 * y0 / h),
            round(1000 * x1 / w), round(1000 * y1 / h)]


def main():
    t0 = time.time()
    print("Loading processor + model...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(CKPT)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Loaded on {device} in {time.time()-t0:.0f}s")

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
        n_words = len(words_list)

        texts = [x["text"] for x in words_list]
        bboxes_norm = [clean_bbox(x["bbox"], w, h) for x in words_list]

        # Chunk: CHUNK_SIZE words per chunk, no overlap
        final_labels = [None] * n_words
        final_scores = [0.0] * n_words

        chunks_info = []
        for start in range(0, n_words, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_words)
            chunk_words = texts[start:end]
            chunk_boxes = bboxes_norm[start:end]

            encoding = processor(
                images=img,
                text=chunk_words,
                boxes=chunk_boxes,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )

            word_ids = encoding.word_ids(batch_index=0)
            model_in = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**model_in)

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            pred_scores_token = torch.max(probs, dim=-1).values.detach().cpu().numpy()

            # Map token predictions → word (majority vote)
            word_pred_ids = {}
            word_pred_scores = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                word_pred_ids.setdefault(word_id, []).append(int(pred_ids[token_idx]))
                word_pred_scores.setdefault(word_id, []).append(float(pred_scores_token[token_idx]))

            for local_idx in range(len(chunk_words)):
                global_idx = start + local_idx
                if local_idx not in word_pred_ids:
                    final_labels[global_idx] = "O"
                    final_scores[global_idx] = 0.0
                else:
                    majority_id = Counter(word_pred_ids[local_idx]).most_common(1)[0][0]
                    final_labels[global_idx] = ID2LABEL.get(majority_id, "O")
                    final_scores[global_idx] = float(np.mean(word_pred_scores[local_idx]))

            chunks_info.append({"start": start, "end": end, "num": end - start})

        final_labels = [l if l is not None else "O" for l in final_labels]

        n_table = sum(1 for l in final_labels if l in ("table", "table_text"))
        results[img_name] = final_labels
        print(f"  {img_name}: {n_words} words, {n_table} table/table_text ({len(chunks_info)} chunks)")

    with open(LABEL_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved labels to {LABEL_OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
