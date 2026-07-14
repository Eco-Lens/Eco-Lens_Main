"""
inference_layoutlmv3.py
------------------------
Load LayoutLMv3 model từ checkpoint fine-tuned.
Inference bằng LayoutLMv3Processor (image + words + boxes).
Chunk words thành các cửa sổ overlap để tránh max_length=512 và giảm lỗi ở biên chunk.

Cách dùng:
    python "2.LayoutLMV3_step/inference_layoutlmv3.py"
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

import transformers.utils.generic as generic
generic.is_tf_available = lambda: False

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

CKPT = os.path.join("Output", "2_Model_Layoutlmv3_Finetune", "checkpoint-1000")
OCR_JSON = os.path.join("test", "output", "step1_ocr", "0_ocr_words.json")
LABEL_OUT = os.path.join("test", "output", "step2_layoutlmv3", "0_layoutlmv3_labels.json")
IMAGE_DIR = "test"

CHUNK_SIZE = 60
CHUNK_STRIDE = 30
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


def _chunk_starts(n_words):
    if n_words <= CHUNK_SIZE:
        return [0]
    starts = list(range(0, n_words - CHUNK_SIZE + 1, CHUNK_STRIDE))
    last_start = n_words - CHUNK_SIZE
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _promote_table_header_labels(words, labels):
    """Recover figure-labelled header groups directly adjacent to table words."""
    table_indices = [i for i, label in enumerate(labels) if label == "table"]
    figure_indices = [i for i, label in enumerate(labels) if label == "figure"]
    if not table_indices or not figure_indices:
        return labels

    heights = [words[i]["bbox"][3] - words[i]["bbox"][1] for i in table_indices]
    median_h = max(1, sorted(heights)[len(heights) // 2])

    # Group nearby figure words first so a multi-column header is assessed as a unit.
    pending = set(figure_indices)
    groups = []
    while pending:
        group = {pending.pop()}
        changed = True
        while changed:
            changed = False
            for candidate in list(pending):
                cb = words[candidate]["bbox"]
                for member in group:
                    mb = words[member]["bbox"]
                    x_gap = max(0, max(cb[0], mb[0]) - min(cb[2], mb[2]))
                    y_gap = max(0, max(cb[1], mb[1]) - min(cb[3], mb[3]))
                    if x_gap <= median_h * 8 and y_gap <= median_h * 3:
                        group.add(candidate)
                        pending.remove(candidate)
                        changed = True
                        break
        groups.append(group)

    for group in groups:
        boxes = [words[i]["bbox"] for i in group]
        group_bbox = [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ]
        adjacent = False
        for table_index in table_indices:
            tb = words[table_index]["bbox"]
            x_gap = max(0, max(group_bbox[0], tb[0]) - min(group_bbox[2], tb[2]))
            y_gap = max(0, max(group_bbox[1], tb[1]) - min(group_bbox[3], tb[3]))
            if x_gap <= median_h * 8 and y_gap <= median_h * 4:
                adjacent = True
                break
        if adjacent:
            for index in group:
                labels[index] = "table"
    return labels


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

        # Average full class probabilities across overlapping windows. This avoids
        # changing a table header label merely because it lies at index 60/120/...
        word_probabilities = [[] for _ in range(n_words)]

        chunks_info = []
        for start in _chunk_starts(n_words):
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
            probs_np = probs.detach().cpu().numpy()

            # Average subword probabilities into each OCR word.
            word_token_probs = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                word_token_probs.setdefault(word_id, []).append(probs_np[token_idx])

            for local_idx in range(len(chunk_words)):
                global_idx = start + local_idx
                if local_idx in word_token_probs:
                    word_probabilities[global_idx].append(
                        np.mean(word_token_probs[local_idx], axis=0)
                    )

            chunks_info.append({"start": start, "end": end, "num": end - start})

        final_labels = []
        final_scores = []
        for predictions in word_probabilities:
            if not predictions:
                final_labels.append("O")
                final_scores.append(0.0)
                continue
            averaged = np.mean(predictions, axis=0)
            label_id = int(np.argmax(averaged))
            final_labels.append(ID2LABEL.get(label_id, "O"))
            final_scores.append(float(averaged[label_id]))

        final_labels = _promote_table_header_labels(words_list, final_labels)

        n_table = sum(1 for l in final_labels if l in ("table", "table_text"))
        results[img_name] = final_labels
        print(f"  {img_name}: {n_words} words, {n_table} table/table_text ({len(chunks_info)} chunks)")

    with open(LABEL_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved labels to {LABEL_OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
