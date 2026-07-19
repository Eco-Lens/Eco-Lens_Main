"""
build_layout_blocks.py — Build document blocks from LayoutLMv3 layout artifact.

Reads the rich ``layout_words.json`` from Step 2 and groups words into
lines → paragraphs → blocks. Does NOT load LayoutLMv3.

Usage:
    python build_layout_blocks.py \\
        --layout-json runs/{run_id}/output/step2_layoutlmv3/layout_words.json \\
        --image-dir runs/{run_id}/pages \\
        --out-json runs/{run_id}/output/step4_semantic_mapping/0_layoutlmv3_layout.json
"""
import sys, os, json, time, re, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter

from pipeline_core.config import SCHEMA_VERSION
from pipeline_core.utils import atomic_write_json, read_json_safe

COLUMN_GAP_RATIO = 0.03
VALID_LABELS = {"text", "figure", "table", "table_text"}


def sort_words_reading_order(words):
    for w in words:
        x0, y0, x1, y1 = w["bbox"]
        w["cx"] = (x0 + x1) / 2
        w["cy"] = (y0 + y1) / 2
    return sorted(words, key=lambda w: (round(w["cy"] / 8), w["bbox"][0]))


def detect_and_split_columns(words):
    if len(words) < 10:
        return [words]
    page_width = 1000
    min_gap = page_width * COLUMN_GAP_RATIO
    flow_labels = {"text", "header", "toc", "figure"}
    flow_words = [w for w in words if w.get("layout_label") in flow_labels]
    if len(flow_words) < 8:
        return [words]
    left_sorted = sorted(flow_words, key=lambda w: w["bbox"][0])
    best_left_gap, left_split = 0, None
    for current, following in zip(left_sorted, left_sorted[1:]):
        gap = following["bbox"][0] - current["bbox"][0]
        if gap > best_left_gap:
            best_left_gap = gap
            left_split = (current["bbox"][0] + following["bbox"][0]) / 2
    if best_left_gap >= min_gap and left_split:
        col_a = [w for w in words if w["bbox"][0] < left_split]
        col_b = [w for w in words if w["bbox"][0] >= left_split]
        if len(col_a) >= 3 and len(col_b) >= 3:
            return [col_a, col_b]
    sorted_flow = sorted(flow_words, key=lambda w: w["bbox"][0])
    max_gap, split_x = 0, None
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
    if not words:
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
    if not lines:
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
        line_height = np.mean([w["bbox"][3] - w["bbox"][1] for w in prev])
        gap_threshold = max(3, min(line_height * line_gap_ratio, typical_gap * 2.5 + 2))
        page_width = 1000
        indent_threshold = page_width * indent_ratio
        prev_bottom = max(w["bbox"][3] for w in prev)
        line_top = min(w["bbox"][1] for w in line)
        gap = line_top - prev_bottom
        prev_left = min(w["bbox"][0] for w in prev)
        line_left = min(w["bbox"][0] for w in line)
        prev_text = " ".join(w.get("text", "") for w in sorted(prev, key=lambda x: x["bbox"][0])).strip()
        line_text = " ".join(w.get("text", "") for w in sorted(line, key=lambda x: x["bbox"][0])).strip()
        explicit_scope_start = re.match(r"^[\s:;,.•·\-–—>*]*scope\s*[123]\b", line_text, re.IGNORECASE)
        explicit_start = line_text.startswith((">", "•", "- ")) or bool(explicit_scope_start)
        completed_sentence = prev_text.endswith((".", "!", "?", ":"))
        paragraph_break = explicit_start or (completed_sentence and gap > typical_gap + 1)
        if gap <= gap_threshold and abs(prev_left - line_left) <= indent_threshold and not paragraph_break:
            current.append(line)
        else:
            paragraphs.append(current)
            current = [line]
    paragraphs.append(current)
    return paragraphs


def group_text_blocks(words, page_width=0, page_height=0):
    filtered = [w for w in words if w.get("layout_label") in VALID_LABELS]
    if not filtered:
        return []
    all_blocks = []
    prose_words = [w for w in filtered if w.get("layout_label") in {"text", "figure"}]
    table_words = [w for w in filtered if w.get("layout_label") in {"table", "table_text"}]
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
                text_lines.append(" ".join(w.get("text", "") for w in line_sorted))
            text = "\n".join(text_lines)
            bbox = [
                min(w["bbox"][0] for w in para_words),
                min(w["bbox"][1] for w in para_words),
                max(w["bbox"][2] for w in para_words),
                max(w["bbox"][3] for w in para_words),
            ]
            confidence = float(np.mean([w.get("layout_confidence", 0) for w in para_words]))
            label_counts = Counter(w.get("layout_label", "O") for w in para_words)
            dominant_label = label_counts.most_common(1)[0][0]
            all_blocks.append({
                "type": dominant_label,
                "labels": sorted(label_counts.keys()),
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "num_lines": len(para),
                "num_words": len(para_words),
                "words": [
                    {"text": w.get("text", ""), "bbox": w["bbox"],
                     "label": w.get("layout_label", "O"),
                     "confidence": w.get("layout_confidence", 0)}
                    for w in para_words
                ],
            })
    return all_blocks


def main():
    ap = argparse.ArgumentParser(description="Build document blocks from layout artifact")
    ap.add_argument("--layout-json", required=True, help="Rich layout_words.json from Step 2")
    ap.add_argument("--image-dir", default=None, help="Page images directory (for dimension only)")
    ap.add_argument("--out-json", required=True, help="Output layout blocks JSON")
    ap.add_argument("--run-id", default=None, help="Run ID (for metadata)")
    args = ap.parse_args()

    t0 = time.time()
    run_id = args.run_id or os.environ.get("RUN_ID", "unknown")

    print(f"Reading layout artifact: {args.layout_json}")
    artifact = read_json_safe(args.layout_json)
    if not artifact:
        print("ERROR: layout artifact not found or invalid")
        sys.exit(1)

    pages = artifact.get("pages", {})
    if not pages:
        print("ERROR: no pages in layout artifact")
        sys.exit(1)

    print(f"Found {len(pages)} pages in artifact")

    results = {}
    for img_name, page_data in sorted(pages.items()):
        words = page_data.get("words", [])
        if not words:
            results[img_name] = {
                "page": img_name, "num_words": 0, "num_blocks": 0,
                "block_summary": {}, "blocks": [],
            }
            continue

        blocks = group_text_blocks(words)
        block_counter = Counter(b["type"] for b in blocks)
        results[img_name] = {
            "page": img_name,
            "num_words": len(words),
            "num_blocks": len(blocks),
            "block_summary": dict(block_counter),
            "blocks": blocks,
        }
        print(f"  {img_name}: {len(blocks)} blocks {dict(block_counter)}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    atomic_write_json(results, args.out_json, schema_version=SCHEMA_VERSION)
    print(f"\nSaved layout blocks to {args.out_json}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
