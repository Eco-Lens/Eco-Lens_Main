"""
inference_layoutlmv3.py
-----------------------
Load LayoutLMv3 model từ checkpoint fine-tuned.
Inference bằng LayoutLMv3Processor (image + words + boxes).
Chunk words thành các cửa sổ overlap để tránh max_length=512 và lỗi biên chunk.

Cách dùng:
    python "4.SemanticMapping/layoutlmv3_inference.py"
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
OCR_JSON = os.path.join("test", "output", "step1_ocr", "0_ocr_words.json")
LAYOUT_JSON_OUT = os.path.join("test", "output", "step4_semantic_mapping", "0_layoutlmv3_layout.json")
IMAGE_DIR = "test"

CHUNK_SIZE = 60
CHUNK_STRIDE = 30
MAX_LEN = 512

ID2LABEL = {
    0: "O", 1: "chart", 2: "figure", 3: "footer", 4: "header",
    5: "ignore", 6: "table", 7: "table_text", 8: "text", 9: "toc",
}

COLUMN_GAP_RATIO = 0.03


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


def sort_words_reading_order(words):
    for w in words:
        x0, y0, x1, y1 = w["bbox"]
        w["cx"] = (x0 + x1) / 2
        w["cy"] = (y0 + y1) / 2
    return sorted(words, key=lambda w: (round(w["cy"] / 8), w["bbox"][0]))


def detect_and_split_columns(words):
    """
    Detect multi-column layout by analyzing x-gaps between text-flow words.
    Uses center_x gap between consecutive tokens sorted by x,
    excluding scattered elements (chart, table_text) from gap detection.
    Returns list of word groups, one per column.
    """
    if len(words) < 10:
        return [words]

    page_width = 1000
    min_gap = page_width * COLUMN_GAP_RATIO  # 1000 * 0.08 = 80

    # Only use flowing-text labels for column detection
    flow_labels = {"text", "header", "toc", "figure"}
    flow_words = [w for w in words if w["label"] in flow_labels]
    if len(flow_words) < 8:
        return [words]

    # Repeated left edges are more reliable than a page-wide whitespace gutter:
    # a page can switch from two columns to one full-width section lower down.
    left_sorted = sorted(flow_words, key=lambda w: w["bbox"][0])
    best_left_gap = 0
    left_split = None
    for current, following in zip(left_sorted, left_sorted[1:]):
        gap = following["bbox"][0] - current["bbox"][0]
        if gap > best_left_gap:
            best_left_gap = gap
            left_split = (current["bbox"][0] + following["bbox"][0]) / 2
    if best_left_gap >= min_gap and left_split is not None:
        col_a = [w for w in words if w["bbox"][0] < left_split]
        col_b = [w for w in words if w["bbox"][0] >= left_split]
        if len(col_a) >= 3 and len(col_b) >= 3:
            return [col_a, col_b]

    # Fall back to a true whitespace gutter when left edges are less regular.
    sorted_flow = sorted(flow_words, key=lambda w: w["bbox"][0])

    # Find the largest gap between consecutive right_edge -> left_edge
    max_gap = 0
    split_x = None
    right_edge = sorted_flow[0]["bbox"][2]
    for i in range(len(sorted_flow) - 1):
        curr_right = right_edge
        next_left = sorted_flow[i + 1]["bbox"][0]
        gap = next_left - curr_right
        if gap > max_gap:
            max_gap = gap
            split_x = (curr_right + next_left) / 2
        right_edge = max(right_edge, sorted_flow[i + 1]["bbox"][2])

    if max_gap < min_gap or split_x is None:
        return [words]

    col_a = [w for w in words if w["bbox"][0] < split_x]
    col_b = [w for w in words if w["bbox"][0] >= split_x]

    if len(col_a) < 3 or len(col_b) < 3:
        return [words]

    return [col_a, col_b]


def reconstruct_lines(words, y_threshold_ratio=0.5, h_gap_ratio=2.0):

    if len(words) == 0:
        return []

    words = sort_words_reading_order(words)

    heights = [w["bbox"][3] - w["bbox"][1] for w in words]
    widths = [w["bbox"][2] - w["bbox"][0] for w in words]
    median_h = np.median(heights) if heights else 20
    median_w = np.median(widths) if widths else 10

    y_threshold = max(5, median_h * y_threshold_ratio)
    h_gap_threshold = median_w * h_gap_ratio

    lines = []
    current = [words[0]]
    current_y = (words[0]["bbox"][1] + words[0]["bbox"][3]) / 2

    for w in words[1:]:
        same_line = abs(w["cy"] - current_y) <= y_threshold
        if same_line:
            current.append(w)
            current_y = np.mean([x["cy"] for x in current])
        else:
            current.sort(key=lambda x: x["bbox"][0])
            lines.append(current)
            current = [w]
            current_y = w["cy"]

    current.sort(key=lambda x: x["bbox"][0])
    lines.append(current)

    # Split each line by horizontal gap
    split_lines = []
    for line in lines:
        segments = []
        seg = [line[0]]
        for w in line[1:]:
            prev_right = seg[-1]["bbox"][2]
            curr_left = w["bbox"][0]
            if curr_left - prev_right > h_gap_threshold:
                segments.append(seg)
                seg = [w]
            else:
                seg.append(w)
        segments.append(seg)
        split_lines.extend(segments)

    return split_lines


def merge_lines_to_paragraphs(lines, line_gap_ratio=0.9, indent_ratio=0.04):

    if len(lines) == 0:
        return []

    line_gaps = []
    for prev, line in zip(lines, lines[1:]):
        gap = min(w["bbox"][1] for w in line) - max(w["bbox"][3] for w in prev)
        if gap >= 0:
            line_gaps.append(gap)
    typical_gap = float(np.median(line_gaps)) if line_gaps else 2.0

    paragraphs = []
    current = [lines[0]]

    for line in lines[1:]:

        prev = current[-1]

        line_height = np.mean([
            w["bbox"][3] - w["bbox"][1]
            for w in prev
        ])

        line_gap = max(3, min(line_height * line_gap_ratio, typical_gap * 2.5 + 2))
        page_width = 1000
        indent_threshold = page_width * indent_ratio

        prev_bottom = max(w["bbox"][3] for w in prev)
        line_top = min(w["bbox"][1] for w in line)
        gap = line_top - prev_bottom

        prev_left = min(w["bbox"][0] for w in prev)
        line_left = min(w["bbox"][0] for w in line)

        prev_text = " ".join(w["text"] for w in sorted(prev, key=lambda x: x["bbox"][0])).strip()
        line_text = " ".join(w["text"] for w in sorted(line, key=lambda x: x["bbox"][0])).strip()
        explicit_start = line_text.startswith((">", "•", "- "))
        completed_sentence = prev_text.endswith((".", "!", "?", ":"))
        paragraph_break = explicit_start or (completed_sentence and gap > typical_gap + 1)

        if (gap <= line_gap and abs(prev_left - line_left) <= indent_threshold
                and not paragraph_break):
            current.append(line)
        else:
            paragraphs.append(current)
            current = [line]

    paragraphs.append(current)
    return paragraphs


def group_text_blocks(words):
    """
    Group OCR words into text/figure/header/toc blocks.
    Includes multi-column detection to avoid merging separate columns.
    """
    # Include meaningful labels for downstream use
    valid_labels = {"text", "figure", "table", "table_text"}

    filtered = [w for w in words if w["label"] in valid_labels]
    if len(filtered) == 0:
        return []

    all_blocks = []
    prose_words = [w for w in filtered if w["label"] in {"text", "figure"}]
    table_words = [w for w in filtered if w["label"] in {"table", "table_text"}]
    word_groups = detect_and_split_columns(prose_words) if prose_words else []
    if table_words:
        word_groups.append(table_words)

    for col_words in word_groups:
        col_words = sort_words_reading_order(col_words)
        lines = reconstruct_lines(col_words)
        paragraphs = merge_lines_to_paragraphs(lines)

        for para in paragraphs:
            para_words = []
            para.sort(key=lambda line: min(w["bbox"][1] for w in line))
            for line in para:
                line.sort(key=lambda x: x["bbox"][0])
                para_words.extend(line)

            text_lines = []
            for line in para:
                line_sorted = sorted(line, key=lambda x: x["bbox"][0])
                text_lines.append(" ".join(w["text"] for w in line_sorted))
            text = "\n".join(text_lines)

            bbox = [
                min(w["bbox"][0] for w in para_words),
                min(w["bbox"][1] for w in para_words),
                max(w["bbox"][2] for w in para_words),
                max(w["bbox"][3] for w in para_words),
            ]

            confidence = float(np.mean([w["confidence"] for w in para_words]))

            # Determine dominant label (most frequent, not just set)
            label_counts = Counter(w["label"] for w in para_words)
            dominant_label = label_counts.most_common(1)[0][0]

            all_blocks.append({
                "type": dominant_label,
                "labels": sorted(label_counts.keys()),
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "num_lines": len(para),
                "num_words": len(para_words),
                "words": para_words,
            })

    return all_blocks


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
            results[img_name] = {
                "page": img_name,
                "num_words": 0,
                "num_blocks": 0,
                "block_summary": {},
                "blocks": [],
            }
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            results[img_name] = {
                "page": img_name,
                "num_words": 0,
                "num_blocks": 0,
                "block_summary": {},
                "blocks": [],
            }
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        n_words = len(words_list)

        texts = [x["text"] for x in words_list]
        bboxes_norm = [clean_bbox(x["bbox"], w, h) for x in words_list]

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

        # Relabel: words marked "figure" but spatially adjacent to "table" regions
        # are likely table headers → change to "table_text"
        table_bboxes = [
            bboxes_norm[i] for i in range(n_words)
            if final_labels[i] in ("table", "table_text")
        ]
        if table_bboxes:
            table_x_min = min(b[0] for b in table_bboxes)
            table_x_max = max(b[2] for b in table_bboxes)
            table_y_min = min(b[1] for b in table_bboxes)
            table_y_max = max(b[3] for b in table_bboxes)
            table_y_mid = (table_y_min + table_y_max) / 2
            for i in range(n_words):
                if final_labels[i] == "figure":
                    b = bboxes_norm[i]
                    bx_overlap = min(b[2], table_x_max) - max(b[0], table_x_min)
                    by_dist = abs((b[1] + b[3]) / 2 - table_y_mid)
                    if bx_overlap > 0 and by_dist < 300:
                        final_labels[i] = "table_text"

        word_objects = []
        for i in range(n_words):
            word_objects.append({
                "text": texts[i],
                "bbox": bboxes_norm[i],
                "label": final_labels[i],
                "confidence": final_scores[i],
            })

        blocks = group_text_blocks(word_objects)

        block_counter = Counter(block["type"] for block in blocks)

        results[img_name] = {
            "page": img_name,
            "num_words": len(word_objects),
            "num_blocks": len(blocks),
            "block_summary": dict(block_counter),
            "blocks": blocks,
        }
        print(
            f"{img_name}: "
            f"{len(blocks)} blocks "
            f"{dict(block_counter)} "
            f"({len(chunks_info)} chunks)"
        )

    with open(LAYOUT_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved layout json to {LAYOUT_JSON_OUT}")


if __name__ == "__main__":
    main()
