"""
inference_layoutlmv3.py — LayoutLMv3 word-level label inference.

Produces two artifacts:
  1. Legacy labels:  {filename: [label_str, ...]}  (backward compat)
  2. Rich layout:    pipeline_core.layout_words.json with per-word scores.

Usage:
    python inference_layoutlmv3.py \\
        --ocr-json runs/{run_id}/output/step1_ocr/ocr_words.json \\
        --image-dir runs/{run_id}/pages \\
        --out-labels runs/{run_id}/output/step2_layoutlmv3/0_layoutlmv3_labels.json \\
        --out-layout runs/{run_id}/output/step2_layoutlmv3/layout_words.json
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from PIL import Image

import transformers.utils.generic as generic
generic.is_tf_available = lambda: False

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from pipeline_core.config import LAYOUT_LABELS, SCHEMA_VERSION
from pipeline_core.context import RunContext
from pipeline_core.utils import atomic_write_json

CHUNK_SIZE = 60
CHUNK_STRIDE = 30
MAX_LEN = 512


def clean_bbox(bbox, w, h):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
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
    table_indices = [i for i, label in enumerate(labels) if label == "table"]
    figure_indices = [i for i, label in enumerate(labels) if label == "figure"]
    if not table_indices or not figure_indices:
        return labels
    heights = [words[i]["bbox"][3] - words[i]["bbox"][1] for i in table_indices]
    median_h = max(1, sorted(heights)[len(heights) // 2])
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
        for table_index in table_indices:
            tb = words[table_index]["bbox"]
            x_gap = max(0, max(group_bbox[0], tb[0]) - min(group_bbox[2], tb[2]))
            y_gap = max(0, max(group_bbox[1], tb[1]) - min(group_bbox[3], tb[3]))
            if x_gap <= median_h * 8 and y_gap <= median_h * 4:
                for index in group:
                    labels[index] = "table"
                break
    return labels


def main():
    ap = argparse.ArgumentParser(description="LayoutLMv3 word-level label inference")
    ap.add_argument("--ocr-json", required=True, help="OCR words JSON")
    ap.add_argument("--image-dir", required=True, help="Page images directory")
    ap.add_argument("--out-labels", required=True, help="Legacy labels JSON output")
    ap.add_argument("--out-layout", required=True, help="Rich layout artifact output")
    ap.add_argument("--ckpt", default=None, help="LayoutLMv3 checkpoint path")
    ap.add_argument("--run-id", default=None, help="Run ID (for metadata)")
    args = ap.parse_args()

    t0 = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = args.ckpt or os.path.join(project_root, "Output", "2_Model_Layoutlmv3_Finetune", "checkpoint-1000")
    run_id = args.run_id or os.environ.get("RUN_ID", "unknown")

    print("Loading processor + model...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(ckpt)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Loaded on {device} in {time.time()-t0:.0f}s")

    with open(args.ocr_json, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    legacy_labels = {}
    rich_pages = {}

    for img_name, words_list in sorted(ocr_data.items()):
        if not words_list:
            legacy_labels[img_name] = []
            rich_pages[img_name] = {"width": 0, "height": 0, "words": []}
            continue

        img_path = os.path.join(args.image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_name} not found")
            legacy_labels[img_name] = []
            rich_pages[img_name] = {"width": 0, "height": 0, "words": []}
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        n_words = len(words_list)
        texts = [x["text"] for x in words_list]
        bboxes_norm = [clean_bbox(x["bbox"], w, h) for x in words_list]

        word_probabilities = [[] for _ in range(n_words)]
        for start in _chunk_starts(n_words):
            end = min(start + CHUNK_SIZE, n_words)
            chunk_words = texts[start:end]
            chunk_boxes = bboxes_norm[start:end]
            encoding = processor(
                images=img, text=chunk_words, boxes=chunk_boxes,
                truncation=True, padding="max_length",
                max_length=MAX_LEN, return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            model_in = {k: v.to(device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = model(**model_in)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            word_token_probs = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                word_token_probs.setdefault(word_id, []).append(probs[token_idx])
            for local_idx in range(len(chunk_words)):
                global_idx = start + local_idx
                if local_idx in word_token_probs:
                    word_probabilities[global_idx].append(
                        np.mean(word_token_probs[local_idx], axis=0)
                    )

        final_labels = []
        final_scores = []
        all_probs = []
        for predictions in word_probabilities:
            if not predictions:
                final_labels.append("O")
                final_scores.append(0.0)
                all_probs.append(None)
                continue
            averaged = np.mean(predictions, axis=0)
            label_id = int(np.argmax(averaged))
            final_labels.append(LAYOUT_LABELS.get(label_id, "O"))
            final_scores.append(float(averaged[label_id]))
            all_probs.append({LAYOUT_LABELS.get(i, "O"): round(float(averaged[i]), 4)
                              for i in range(len(LAYOUT_LABELS))})

        final_labels = _promote_table_header_labels(words_list, final_labels)

        legacy_labels[img_name] = final_labels
        rich_pages[img_name] = {
            "width": w,
            "height": h,
            "words": [
                {
                    "index": i,
                    "text": words_list[i]["text"],
                    "bbox": bboxes_norm[i],
                    "ocr_confidence": words_list[i].get("conf", 0),
                    "layout_label": final_labels[i],
                    "layout_confidence": final_scores[i],
                }
                for i in range(n_words)
            ],
        }
        n_table = sum(1 for l in final_labels if l in ("table", "table_text"))
        print(f"  {img_name}: {n_words} words, {n_table} table/table_text")

    os.makedirs(os.path.dirname(args.out_labels), exist_ok=True)
    atomic_write_json(legacy_labels, args.out_labels, schema_version=SCHEMA_VERSION)

    layout_artifact = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": {
            "name": "layoutlmv3",
            "checkpoint": ckpt,
            "labels": LAYOUT_LABELS,
        },
        "pages": rich_pages,
    }
    atomic_write_json(layout_artifact, args.out_layout)

    elapsed = time.time() - t0
    print(f"\nSaved labels to {args.out_labels}")
    print(f"Saved layout artifact to {args.out_layout}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
