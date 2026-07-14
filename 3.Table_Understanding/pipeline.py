import sys, os, json, time, re, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw, ImageFont
import torch; import torchvision

from config import *
from utils import (
    merge_bboxes, bbox_area, expand_bbox,
    split_by_columns, split_by_vertical_gap, split_virtual_panels,
    merge_row_aligned_groups,
    is_paragraph_cell, is_numeric, is_numeric_lenient, parse_number,
)
from tatr_engine import TATREngine

import re

INVALID_METRICS = {
    "",
    "item",
    "items",
    "unit",
    "units",
    "%",
    "times",
    "description",
    "year",
    "years",
    "no",
    "no.",
}

SECTION_WORDS = {
    "assets",
    "liabilities",
    "equity",
    "capital",
    "owner",
    "shareholders",
    "balance sheet",
    "cash flow",
    "income statement",
}

METRIC_KEYWORDS = {

    "metric",
    "indicator",
    "description",
    "item",
    "particular",
    "particulars",
    "category",
    "parameter",
    "measure",
    "kpi",
    "building",
}

UNIT_KEYWORDS = {

    "unit",
    "units",
    "uom",
}
TOTAL_WORDS = {
    "total",
    "subtotal",
    "grand total",
}

INVALID_METRICS = {
    "",
    "item",
    "items",
    "unit",
    "units",
    "%",
    "times",
    "description",
    "year",
    "years",
    "no",
    "no.",
}

HEADER_KEYWORDS = {
    "metric",
    "indicator",
    "description",
    "item",
    "note",
    "notes",
    "unit",
    "year",
    "scope",
    "category",
    "kpi",
} 

def normalize_text(text: str):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_number(text):
    if text is None:
        return False

    text = str(text).strip()

    if text == "":
        return False

    text = text.replace(",", "")

    try:
        float(text)
        return True
    except:
        return False
    
def is_heading(metric):

    metric = normalize_text(metric)

    if not metric:
        return False

    lower = metric.lower()

    if any(k in lower for k in SECTION_WORDS):
        return True

    letters = [c for c in metric if c.isalpha()]

    if letters:

        ratio = sum(c.isupper() for c in letters) / len(letters)

        if ratio > 0.8:
            return True

    return False

def extract_year(text):

    if text is None:
        return None

    m = re.search(r"(19|20)\d{2}", str(text))

    if m:
        return int(m.group())

    return None

import re

def is_total(metric):

    metric = metric.lower()

    return any(k in metric for k in TOTAL_WORDS)

def normalize_text(text: str):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_metric(metric):

    metric = normalize_text(metric)

    metric = metric.replace("\n"," ")

    metric = re.sub(r"\s+"," ",metric)

    return metric.strip()

def is_number(text):
    if text is None:
        return False

    text = str(text).strip()

    if text == "":
        return False

    text = text.replace(",", "")

    try:
        float(text)
        return True
    except:
        return False


def extract_year(text):

    if text is None:
        return None

    m = re.search(r"(19|20)\d{2}", str(text))

    if m:
        return int(m.group())

    return None

def detect_header_row(table_grid):

    best_row = 0
    best_score = -1

    for ri, row in enumerate(table_grid):

        score = 0
        non_empty = [c for c in row if normalize_text(c)]

        if len(non_empty) <= 1:
            continue
        for cell in row:
            cell = normalize_text(cell).lower()
            header_hits = 0
            if extract_year(cell):

                header_hits += 1

            elif any(k in cell for k in HEADER_KEYWORDS):

                header_hits += 1
            score += header_hits * 3
            if not cell:
                continue

            lower = cell.lower()

            # keyword
            if any(k in lower for k in HEADER_KEYWORDS):
                score += 5

            # year
            if extract_year(cell):
                score += 3

            # text
            if not is_number(cell):
                score += 1

            # numeric data
            else:
                score -= 1

        if score > best_score:
            best_score = score
            best_row = ri

    return best_row

def detect_columns(header):

    metric_col = None

    unit_col = None

    year_cols = {}

    text_score = []

    for ci, cell in enumerate(header):

        txt = normalize_text(cell)

        lower = txt.lower()

        score = 0

        # metric keyword
        if any(k in lower for k in METRIC_KEYWORDS):
            score += 100

        # year column
        year = extract_year(txt)

        if year:
            year_cols[ci] = year
            score -= 100

        # unit
        if any(k in lower for k in UNIT_KEYWORDS):
            unit_col = ci
            score -= 50

        # text-like header
        if txt and not is_number(txt):
            score += 5

        text_score.append(score)

    if text_score:

        metric_col = max(
            range(len(text_score)),
            key=lambda i: text_score[i]
        )

    else:

        metric_col = 0

    return metric_col, unit_col, year_cols


def _include_adjacent_table_headers(bbox, table_tokens, words, labels):
    """Extend a table seed upward to include nearby figure/table_text headers."""
    heights = [t["bbox"][3] - t["bbox"][1] for t in table_tokens]
    median_h = max(1, sorted(heights)[len(heights) // 2])
    candidates = []
    for word, label in zip(words, labels):
        if label not in {"figure", "table_text"}:
            continue
        wb = word["bbox"]
        x_overlap = max(0, min(wb[2], bbox[2]) - max(wb[0], bbox[0]))
        vertical_gap = max(0, bbox[1] - wb[3])
        if x_overlap > 0 and wb[1] <= bbox[3] and vertical_gap <= median_h * 4:
            candidates.append(wb)
    return merge_bboxes([bbox] + candidates) if candidates else bbox


def _tokens_inside_crop(words, labels, crop_box):
    """Use LayoutLMv3 table labels as seeds, then recover OCR by geometry."""
    tokens = []
    for index, (word, label) in enumerate(zip(words, labels)):
        bbox = word["bbox"]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        if not (crop_box[0] <= center_x <= crop_box[2]
                and crop_box[1] <= center_y <= crop_box[3]):
            continue
        tokens.append({
            "text": word["text"],
            "label": label,
            "ocr_index": index,
            "bbox": [
                bbox[0] - crop_box[0], bbox[1] - crop_box[1],
                bbox[2] - crop_box[0], bbox[3] - crop_box[1],
            ],
        })
    return tokens


def _region_has_table_header(region):
    texts = [normalize_text(t["text"]).lower() for t in region["tokens"]]
    year_count = sum(1 for text in texts if extract_year(text))
    keyword_count = sum(
        1 for text in texts
        if any(keyword in text for keyword in HEADER_KEYWORDS | METRIC_KEYWORDS)
    )
    return year_count >= 2 or keyword_count >= 1


def _group_continuation_regions(regions, iw, ih):
    """Join bottom-to-top continuations across virtual pages, retaining segments."""
    if not regions:
        return []
    panel_count = max(r["panel"] for r in regions) + 1
    panel_width = iw / panel_count
    groups = [[regions[0]]]
    for region in regions[1:]:
        previous = groups[-1][-1]
        previous_width = (previous["bbox"][2] - previous["bbox"][0]) / panel_width
        current_width = (region["bbox"][2] - region["bbox"][0]) / panel_width
        continuation = (
            region["panel"] == previous["panel"] + 1
            and previous["bbox"][3] >= ih * 0.75
            and region["bbox"][1] <= ih * 0.30
            and _region_has_table_header(previous)
            and not _region_has_table_header(region)
            and abs(previous_width - current_width) <= 0.25
        )
        if continuation:
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def _combine_table_segments(segment_results, page, table_id):
    max_cols = max(result["cols"] for result in segment_results)
    table_data = []
    segments = []
    for result in segment_results:
        table_data.extend([
            row + [""] * (max_cols - len(row))
            for row in result["table_data"]
        ])
        segments.append(result["segment"])

    extracted = _extract_esg(table_data, page=page, table_id=table_id)
    has_esg = any(
        keyword in cell.lower()
        for row in table_data for cell in row for keyword in ESG_KEYWORDS
    )
    first = segment_results[0]
    return {
        "page": page,
        "table_id": table_id,
        # A multipart table has no truthful rectangular union. Keep the first
        # segment here for legacy consumers and expose all geometry explicitly.
        "bbox": segments[0]["bbox"],
        "layout_bbox": segments[0]["bbox"],
        "crop_bbox": segments[0]["crop_bbox"],
        "segments": segments,
        "source_label": TABLE_LABEL,
        "table_data": table_data,
        "rows": len(table_data),
        "cols": max_cols,
        "extracted_metrics": extracted,
        "is_esg": has_esg,
        "tatr_cells": sum(result["tatr_cells"] for result in segment_results),
        "tatr_time": round(sum(result["tatr_time"] for result in segment_results), 2),
        "tokens_in_crop": sum(result["tokens_in_crop"] for result in segment_results),
    }

def run(ocr_path, labels_path, image_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    tatr = TATREngine()
    all_results = []

    for img_name in sorted(ocr_data.keys()):
        print("="*60)
        print(img_name)

        words = ocr_data.get(img_name, [])
        labels = label_data.get(img_name, [])

        if len(words) != len(labels):
            raise ValueError(
                f"OCR/label length mismatch for {img_name}: {len(words)} != {len(labels)}"
            )

        print("words :", len(words))
        print("labels:", len(labels))

        img_path = os.path.join(image_root, img_name)
        print("image exists:", os.path.exists(img_path))
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        page_area = iw * ih
        base = os.path.splitext(img_name)[0]
        page_dir = os.path.join(out_dir, base)
        os.makedirs(page_dir, exist_ok=True)

        # Output 1: OCR visualization
        _draw_ocr_vis(img, words, page_dir, base)

        page_log = {"page": base, "n_table_tokens": 0, "n_regions": 0, "regions": [], "skipped": []}

        # Filter table tokens
        table_tokens = [
            {
                "text": words[i]["text"],
                "bbox": words[i]["bbox"],
                "label": labels[i],
                "ocr_index": i,
            }
            for i in range(min(len(words), len(labels)))
            if labels[i] == TABLE_LABEL
        ]
        print(img_name)
        print("OCR words:", len(words))
        print("Labels:", len(labels))
        print("Table tokens:", len(table_tokens))
        # Output 2: LayoutLMv3 label visualization
        _draw_layoutlmv3_vis(img, words, labels, page_dir, base)

        page_results = []
        page_log["n_table_tokens"] = len(table_tokens)

        if len(table_tokens) < MIN_TOKENS_PER_REGION:
            _export_empty(page_dir, base)
            print(f"  {img_name}: 0 tables (<{MIN_TOKENS_PER_REGION} table tokens)")
            all_results.extend(page_log["regions"])
            continue
        print("MIN:", MIN_TOKENS_PER_REGION)
        # Split a two-page spread before vertical clustering. Otherwise rows from
        # the opposite page fill every vertical gap and unrelated tables merge.
        panel_groups = split_virtual_panels(table_tokens, iw, ih)

        # Build physical table regions inside each virtual page/panel.
        valid_regions = []
        print("Virtual panels:", len(panel_groups))
        print("Valid regions:", len(valid_regions))
        for panel_index, panel_tokens in panel_groups:
            vertical_clusters = split_by_vertical_gap(
                panel_tokens, ih, VERTICAL_GAP_MULTIPLIER
            )
            for cluster in vertical_clusters:
                if len(cluster) < MIN_TOKENS_PER_REGION:
                    page_log["skipped"].append({"reason": "too_few_table_tokens", "n": len(cluster)})
                    continue

                col_groups = merge_row_aligned_groups(
                    split_by_columns(cluster, iw, COLUMN_GAP_RATIO)
                )
                for group in col_groups:
                    if len(group) < MIN_TOKENS_PER_REGION:
                        continue

                    bbox = _include_adjacent_table_headers(
                        merge_bboxes([t["bbox"] for t in group]),
                        group, words, labels,
                    )
                    crop_box = expand_bbox(bbox, EXPAND_MARGIN, EXPAND_MARGIN, iw, ih)

                    area_ratio = bbox_area(crop_box) / page_area
                    if area_ratio > MAX_TABLE_REGION_AREA_RATIO:
                        page_log["skipped"].append({"reason": "region_too_large", "bbox": crop_box, "area_ratio": area_ratio})
                        continue

                    num_inside = sum(1 for t in group if is_numeric_lenient(t["text"]))
                    if num_inside < 2:
                        page_log["skipped"].append({"reason": "no_numeric_content", "bbox": crop_box})
                        continue

                    valid_regions.append({
                        "bbox": bbox, "crop_box": crop_box, "tokens": group,
                        "panel": panel_index,
                    })

        valid_regions.sort(key=lambda r: (r["panel"], r["bbox"][1], r["bbox"][0]))
        logical_regions = _group_continuation_regions(valid_regions, iw, ih)
        page_log["n_regions"] = len(logical_regions)

        # TATR runs on each physical segment, never on the envelope of a
        # discontinuous continuation table.
        for ti, region_group in enumerate(logical_regions):
            table_id = f"table_{ti:02d}"
            segment_results = []
            for part_index, reg in enumerate(region_group):
                crop_box = reg["crop_box"]
                crop_img = img.crop(crop_box)
                crop_tokens = _tokens_inside_crop(words, labels, crop_box)
                artifact_id = table_id if len(region_group) == 1 else f"{table_id}_part_{part_index:02d}"

                if len(crop_tokens) < MIN_TOKENS_PER_REGION:
                    page_log["skipped"].append({"reason": "crop_too_few_tokens", "bbox": crop_box})
                    continue

                result, fail_reason = _process_crop(
                    tatr, crop_img, crop_tokens, reg["bbox"], crop_box,
                    base, table_id, artifact_id, page_dir, img,
                )
                if result:
                    segment_results.append(result)
                else:
                    page_log["skipped"].append({"reason": "processing_failed", "detail": fail_reason, "bbox": crop_box})

            if not segment_results:
                continue
            result = _combine_table_segments(segment_results, base, table_id)
            page_results.append(result)
            page_log["regions"].append({
                "table_id": table_id,
                "segments": result["segments"],
                "ocr_words_inside": result["tokens_in_crop"],
                "tatr_cells": result["tatr_cells"],
                "output_shape": [result["rows"], result["cols"]],
            })

        # Draw all table regions on page overview
        _draw_page_overview(img, page_results, page_dir, base)

        # Write log
        def _find_sets(obj, path="page_log"):
            if isinstance(obj, set):
                print(f"!!! SET FOUND at {path}: {obj}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _find_sets(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _find_sets(v, f"{path}[{i}]")

        _find_sets(page_log)

        with open(os.path.join(page_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(page_log, f, ensure_ascii=False, indent=2)


        total_metrics = sum(len(r["extracted_metrics"]) for r in page_results)
        if page_results:
            _export_page_html(page_results, page_dir, base)
        else:
            _export_empty(page_dir, base)

        print(f"  {img_name}: {len(page_results)} tables, {total_metrics} metrics")
        all_results.extend(page_results)

    _export_master(out_dir)
    _export_all_json(all_results, out_dir)
    _export_pending_classification(all_results, out_dir)

    tab_count = sum(1 for r in all_results if r.get("is_esg"))
    print(f"\nDone. {len(all_results)} tables ({tab_count} ESG)")


def _merge_overlapping(regions):
    if len(regions) < 2:
        return regions
    merged = [regions[0]]
    for reg in regions[1:]:
        last = merged[-1]
        lb = last["expanded"]
        rb = reg["expanded"]
        overlap_x = min(lb[2], rb[2]) > max(lb[0], rb[0])
        overlap_y = min(lb[3], rb[3]) > max(lb[1], rb[1])
        if overlap_x and overlap_y:
            merged_bbox = merge_bboxes([last["bbox"], reg["bbox"]])
            merged_expanded = merge_bboxes([lb, rb])
            merged_tokens = last["tokens"] + reg["tokens"]
            merged[-1] = {"bbox": merged_bbox, "expanded": merged_expanded, "tokens": merged_tokens}
        else:
            merged.append(reg)
    return merged


def _process_crop(tatr, crop_img, crop_tokens, raw_bbox, crop_box, base,
                  table_id, artifact_id, page_dir, full_img):
    if len(crop_tokens) < MIN_TOKENS_PER_REGION:
        return None, "too_few_crop_tokens"
    # DEBUG: luôn lưu crop để xem bằng mắt, kể cả khi fail sau này
    os.makedirs(page_dir, exist_ok=True)
    crop_img.save(os.path.join(page_dir, f"DEBUG_{artifact_id}_crop.jpg"))

    t0 = time.time()
    cells = tatr.detect(crop_img)
    det_time = time.time() - t0
    n_rows = sum(1 for c in cells if c["class_name"] == "table row")
    n_cols = sum(1 for c in cells if c["class_name"] == "table column")

    if n_rows < MIN_ROWS or n_cols < MIN_COLS:
        return None, f"tatr_structure_insufficient (rows={n_rows}, cols={n_cols}, MIN_ROWS={MIN_ROWS}, MIN_COLS={MIN_COLS})"

    rows_sorted = sorted(
        [c for c in cells if c["class_name"] == "table row"],
        key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2,
    )
    cols_sorted = sorted(
        [c for c in cells if c["class_name"] == "table column"],
        key=lambda c: (c["bbox"][0] + c["bbox"][2]) / 2,
    )

    grid = {}
    for tok in crop_tokens:
        ri = _best_overlap_index(tok["bbox"], rows_sorted, "y")
        ci = _best_overlap_index(tok["bbox"], cols_sorted, "x")
        if ri is not None and ci is not None:
            grid.setdefault((ri, ci), []).append(tok)

    nrg = len(rows_sorted)
    ncg = len(cols_sorted)
    table_grid = []
    for ri in range(nrg):
        row = []
        for ci in range(ncg):
            toks = grid.get((ri, ci), [])
            toks.sort(key=lambda t: (
                round(((t["bbox"][1] + t["bbox"][3]) / 2) / 8),
                t["bbox"][0],
            ))
            row.append(" ".join(t["text"] for t in toks).strip())
        table_grid.append(row)

    table_grid = _clean_table(table_grid)
    table_grid = _coalesce_continuation_rows(table_grid)
    if len(table_grid) < MIN_ROWS or len(table_grid[0]) < MIN_COLS:
        return None, f"grid_too_small_after_clean ({len(table_grid)}x{len(table_grid[0]) if table_grid else 0})"

    print("Grid:", len(table_grid), len(table_grid[0]))
    extracted = _extract_esg(table_grid, page=base, table_id=table_id)
    has_data = any(is_numeric(cell) for row in table_grid for cell in row)
    has_esg = any(kw in cell.lower() for row in table_grid for cell in row for kw in ESG_KEYWORDS)
    if not has_data:
        return None, "no_numeric_after_tatr_grid"

    _save_debug_images(full_img, crop_img, cells, crop_box, page_dir, artifact_id, base)

    return {
        "page": base, "table_id": table_id, "bbox": raw_bbox,
        "layout_bbox": raw_bbox, "crop_bbox": crop_box, "source_label": TABLE_LABEL,
        "table_data": table_grid, "rows": len(table_grid),
        "cols": len(table_grid[0]) if table_grid else 0,
        "extracted_metrics": extracted, "is_esg": has_esg,
        "tatr_cells": len(cells), "tatr_time": round(det_time, 2),
        "tokens_in_crop": len(crop_tokens),
        "segment": {
            "bbox": raw_bbox,
            "crop_bbox": crop_box,
            "rows": len(table_grid),
            "cols": len(table_grid[0]) if table_grid else 0,
            "artifact_id": artifact_id,
        },
    }, None


def _bbox_overlap_ratio(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return intersection / area


def _best_overlap_index(token_bbox, items, axis="y"):
    token_lo, token_hi = ((token_bbox[1], token_bbox[3]) if axis == "y"
                          else (token_bbox[0], token_bbox[2]))
    token_size = max(1, token_hi - token_lo)
    token_center = (token_lo + token_hi) / 2
    best_i, best_score, best_distance = None, 0.0, float("inf")
    for i, it in enumerate(items):
        lo, hi = (it["bbox"][1], it["bbox"][3]) if axis == "y" else (it["bbox"][0], it["bbox"][2])
        score = max(0, min(token_hi, hi) - max(token_lo, lo)) / token_size
        distance = abs(token_center - (lo + hi) / 2)
        if score > best_score or (score == best_score and distance < best_distance):
            best_i, best_score, best_distance = i, score, distance
    if best_score >= 0.2:
        return best_i
    return best_i if best_distance <= token_size else None


def _clean_table(grid):
    if not grid:
        return []

    # Filter paragraph rows
    cleaned = []
    for row in grid:
        text_cells = [c for c in row if c.strip()]
        if len(text_cells) == 1 and not any(is_numeric_lenient(c) for c in row):
            text = " ".join(text_cells)
            if is_paragraph_cell(text):
                continue
        cleaned.append(row)
    grid = cleaned

    grid = [row for row in grid if any(cell.strip() for cell in row)]

    # Remove trailing empty rows
    while grid and all(not c for c in grid[-1]):
        grid.pop()
    if not grid:
        return []

    # Normalize columns
    ncols = max(len(r) for r in grid)
    for ri in range(len(grid)):
        grid[ri] = grid[ri] + [""] * (ncols - len(grid[ri]))
    while ncols > 0 and all(row[ncols - 1] == "" for row in grid):
        ncols -= 1
        grid = [row[:ncols] for row in grid]
    return grid


def _coalesce_continuation_rows(grid):
    """Merge physical OCR/TATR rows that belong to one logical table row."""
    if len(grid) < 3:
        return grid

    key_col = None
    header_row = None
    key_headers = METRIC_KEYWORDS | {"name", "building"}
    for ri, row in enumerate(grid[:4]):
        for ci, cell in enumerate(row):
            normalized = normalize_text(cell).lower()
            if any(keyword in normalized for keyword in key_headers):
                key_col = ci
                header_row = ri
                break
        if key_col is not None:
            break
    if key_col is None:
        return grid

    def merge_cells(target, source, prepend=False):
        for ci, value in enumerate(source):
            value = value.strip()
            if not value:
                continue
            if target[ci].strip():
                parts = (value, target[ci]) if prepend else (target[ci], value)
                target[ci] = " ".join(parts).strip()
            else:
                target[ci] = value

    row_index = header_row + 1
    while row_index < len(grid):
        row = grid[row_index]
        key = row[key_col].strip()
        previous = grid[row_index - 1] if row_index > header_row + 1 else None
        following = grid[row_index + 1] if row_index + 1 < len(grid) else None

        if not key:
            nonempty = [ci for ci, cell in enumerate(row) if cell.strip()]
            location_only = nonempty and all(ci < key_col for ci in nonempty)
            continues_following_text = bool(following and following[key_col].strip()) and any(
                row[ci].strip() and following[ci].strip()
                and not is_numeric_lenient(row[ci])
                and not is_numeric_lenient(following[ci])
                for ci in range(len(row))
            )
            if following and (location_only or continues_following_text):
                merge_cells(following, row, prepend=True)
                grid.pop(row_index)
                continue
            if previous and previous[key_col].strip():
                merge_cells(previous, row)
                grid.pop(row_index)
                continue

        value_cells = row[key_col + 1:]
        previous_key = previous[key_col].strip() if previous else ""
        is_text_continuation = (
            previous_key.count("(") > previous_key.count(")")
            or previous_key.endswith(("-", ",", "/"))
            or (key[:1].islower() if key else False)
        )
        if (key and not any(cell.strip() for cell in value_cells)
                and previous_key and is_text_continuation):
            merge_cells(previous, row)
            grid.pop(row_index)
            continue

        row_index += 1
    return grid


def _extract_year_column_map(grid, header_rows=2):
    """Quet header_rows dong dau tim nam theo TUNG COT.
    Xu ly bang so sanh nhieu nam: nam chi xuat hien 1 lan o header,
    khong lap lai trong moi cell so lieu."""
    year_by_col = {}
    ncols = len(grid[0]) if grid else 0
    for ri in range(min(header_rows, len(grid))):
        for ci in range(ncols):
            cell = grid[ri][ci].strip()
            if not cell:
                continue
            ym = re.findall(YEAR_PATTERN, cell)
            if ym:
                year_by_col[ci] = int(ym[-1][1])
    return year_by_col


def _extract_esg(table_grid, page, table_id, report_year=None):
    """
    report_year: năm mặc định lấy từ filename/cover page (Mốc học được: nên có
    bước report-year-extraction upstream). Dùng làm fallback khi bảng không có
    cột năm rõ ràng.
    """
    metrics = []
    seen = set()

    if not table_grid:
        return metrics
    if len(table_grid) == 1:
        row = table_grid[0]
        # Heuristic: ô đầu tiên không phải số -> coi là metric label,
        # các ô còn lại nếu là số -> value (year lấy từ report_year, chưa xác định theo cột)
        metric_col = next((ci for ci, c in enumerate(row) if not is_number(c) and c.strip()), None)
        if metric_col is None:
            return metrics  # không có label -> bỏ qua, tránh đoán bừa
        metric = clean_metric(row[metric_col])
        if metric == "" or metric.lower() in INVALID_METRICS or is_heading(metric) or is_total(metric):
            return metrics
        for ci, cell in enumerate(row):
            if ci == metric_col:
                continue
            value = clean_metric(cell)
            if not is_number(value):
                continue
            parsed = parse_number(value)
            if parsed is None:
                continue
            value = parsed
            text = metric.lower()
            metrics.append({
                "metric_id": f"{table_id}_0_{ci}",
                "page": page, "table_id": table_id, "row": 0, "col": ci,
                "metric_text": metric, "value": value, "unit": "",
                "year": report_year,
                "year_source": "single_row_table_report_year",
                "scope": None, "scope_source": "unresolved",
                "needs_classification": True,
                "is_esg": any(kw in text for kw in ESG_KEYWORDS),
            })
        return metrics

    header_row = detect_header_row(table_grid)
    column_header_row = next((
        ri for ri, row in enumerate(table_grid[:4])
        if any(
            any(keyword in normalize_text(cell).lower() for keyword in METRIC_KEYWORDS)
            for cell in row
        )
    ), header_row)
    metric_col, unit_col, year_cols = detect_columns(table_grid[column_header_row])
    data_start_row = max(header_row, column_header_row) + 1

    # --- FALLBACK 1: quét nhiều dòng header thay vì chỉ 1 dòng ---
    if not year_cols:
        year_cols = _extract_year_column_map(table_grid, header_rows=data_start_row)

    # --- FALLBACK 2: bảng chỉ có 1 giá trị/năm (không ghi năm trong bảng) ---
    # Thay vì bỏ toàn bộ bảng, coi mọi cột số (trừ metric_col, unit_col) là
    # "value column" gắn với report_year (có thể None -> "unresolved_year")
    use_fallback_year = False
    if not year_cols:
        use_fallback_year = True
        ncols = len(table_grid[header_row]) if table_grid else 0
        year_cols = {
            ci: report_year
            for ci in range(ncols)
            if ci != metric_col and ci != unit_col
        }

    for ri in range(data_start_row, len(table_grid)):
        row = table_grid[ri]
        if metric_col >= len(row):
            continue

        metric = clean_metric(row[metric_col])
        if metric == "" or metric.lower() in INVALID_METRICS:
            continue
        if is_heading(metric) or is_total(metric):
            continue

        unit = clean_metric(row[unit_col]) if (unit_col is not None and unit_col < len(row)) else ""

        for ci, year in year_cols.items():
            if ci >= len(row):
                continue
            value = clean_metric(row[ci])
            if not is_number(value):
                continue
            parsed = parse_number(value)
            if parsed is None:
                continue
            value = parsed

            key = (metric.lower(), year, value, ci)
            if key in seen:
                continue
            seen.add(key)

            text = (metric + " " + unit).lower()
            is_esg = any(kw in text for kw in ESG_KEYWORDS)

            metrics.append({
                "metric_id": f"{table_id}_{ri}_{ci}",
                "page": page,
                "table_id": table_id,
                "row": ri, "col": ci,
                "metric_text": metric,
                "value": value,
                "unit": unit,
                "year": year,  # có thể None nếu report_year cũng None
                "year_source": "fallback_no_year_column" if use_fallback_year else "table_header",
                "scope": None,
                "scope_source": "unresolved",
                "needs_classification": True,
                "is_esg": is_esg,
            })

    return metrics

def _draw_ocr_vis(img, words, page_dir, base):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    for w in words:
        b = w["bbox"]
        draw.rectangle(b, outline=(0, 0, 255), width=2)
        txt = w["text"]
        if len(txt) > 30:
            txt = txt[:30] + "..."
        ty = max(0, b[1] - 18)
        tb = draw.textbbox((b[0], ty), txt, font=font)
        draw.rectangle(tb, fill=(255, 255, 200))
        draw.text((b[0], ty), txt, fill=(0, 0, 0), font=font)
    vis.save(os.path.join(page_dir, f"{base}_ocr.jpg"))


_LABEL_COLORS = {
    "O": (128, 128, 128), "chart": (255, 165, 0), "figure": (128, 0, 128),
    "footer": (165, 42, 42), "header": (0, 102, 255), "ignore": (180, 180, 180),
    "table": (255, 0, 0), "table_text": (255, 0, 255),
    "text": (0, 180, 0), "toc": (0, 200, 200),
}


def _draw_layoutlmv3_vis(img, words, labels, page_dir, base):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    for i in range(min(len(words), len(labels))):
        b = words[i]["bbox"]
        label = labels[i]
        color = _LABEL_COLORS.get(label, (200, 200, 100))
        draw.rectangle(b, outline=color, width=3)
        ty = max(0, b[1] - 20)
        tb = draw.textbbox((b[0], ty), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((b[0], ty), label, fill=(0, 0, 0), font=font)
    vis.save(os.path.join(page_dir, f"{base}_layoutlmv3.jpg"))


def _draw_page_overview(full_img, page_results, page_dir, base):
    vis = full_img.copy()
    draw = ImageDraw.Draw(vis, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    colors = [(255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 180, 0), (180, 0, 255), (0, 180, 180)]
    for ti, r in enumerate(page_results):
        col = colors[ti % len(colors)]
        for part_index, segment in enumerate(r.get("segments", [])):
            bbox = segment["bbox"]
            draw.rectangle(bbox, outline=col, width=3)
            suffix = f" part {part_index + 1}" if len(r["segments"]) > 1 else ""
            label = f"{r['table_id']}{suffix}"
            draw.text((bbox[0] + 4, max(0, bbox[1] - 20)), label, fill=col, font=font)
    vis.save(os.path.join(page_dir, f"{base}_page_with_table_bboxes.jpg"))


def _save_debug_images(full_img, crop_img, tatr_cells, crop_box, page_dir, artifact_id, base):
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        font = ImageFont.load_default()

    # Crop image
    crop_img.save(os.path.join(page_dir, f"{artifact_id}_crop.jpg"))

    # TATR overlay on crop
    vis2 = crop_img.copy()
    draw2 = ImageDraw.Draw(vis2, "RGBA")
    colors = {
        "table": (220, 50, 50), "table column": (50, 180, 50),
        "table row": (50, 80, 220), "table column header": (220, 180, 30),
        "table projected row header": (180, 30, 180), "table spanning cell": (30, 180, 180),
    }
    for c in tatr_cells:
        col = colors.get(c["class_name"], (128, 128, 128))
        draw2.rectangle(c["bbox"], outline=col, width=2)
        draw2.rectangle(c["bbox"], fill=col + (20,))
    vis2.save(os.path.join(page_dir, f"{artifact_id}_tatr_overlay.jpg"))


def _export_page_html(results, page_dir, base):
    n_tab = len(results)
    n_esg = sum(1 for r in results if r.get("is_esg"))
    total_metrics = sum(len(r["extracted_metrics"]) for r in results)
    nav = []

    for r in results:
        tid = r["table_id"]
        nr = r["rows"]
        nc = r["cols"]
        nm = len(r["extracted_metrics"])
        esg_tag = " ESG" if r.get("is_esg") else ""
        nav.append(
            f'<a href="{tid}.html"><b>Table {int(tid.split("_")[1])+1}</b> ({nr}r x {nc}c, {nm} metrics{esg_tag})</a>'
        )

        html_rows = []
        for ri, row in enumerate(r["table_data"]):
            cells_html = "".join(
                ("<td class='nm'>" if _is_digit(c) else "<td>") + (c if c else chr(8212)) + "</td>"
                for c in row
            )
            html_rows.append(f"<tr><td class='rn'>{ri}</td>{cells_html}</tr>")

        metrics_rows = ""
        for m in r["extracted_metrics"][:50]:
            metrics_rows += (
                f"<tr><td>{m.get('metric_text','')}</td>"
                f"<td class='vl'>{m.get('value','')}</td>"
                f"<td>{m.get('unit','')}</td>"
                f"<td>{m.get('year') or ''}</td>"
                f"<td>{m.get('scope') or ''}</td>"
                f"<td>{'ESG' if m.get('is_esg') else ''}</td></tr>"
            )

        segment_images = "".join(
            f'<div><img src="{segment["artifact_id"]}_crop.jpg">'
            f'<div class="img-label">Table segment {index + 1}</div></div>'
            f'<div><img src="{segment["artifact_id"]}_tatr_overlay.jpg">'
            f'<div class="img-label">TATR segment {index + 1}</div></div>'
            for index, segment in enumerate(r.get("segments", []))
        )

        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><style>
body{{font-family:'Segoe UI',sans-serif;margin:20px;background:#f0f2f4}}
h2{{color:#1a1a2e}}
.tw{{overflow-x:auto;background:white;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.1);margin-bottom:20px}}
table{{border-collapse:collapse;font-size:11px}}
td{{border:1px solid #ddd;padding:4px 8px;white-space:nowrap;max-width:250px;overflow:hidden;text-overflow:ellipsis}}
.nm{{font-family:'Consolas',monospace;text-align:right;font-weight:500;background:#f8fff8}}
.rn{{background:#f0f0f0;color:#999;font-size:10px;text-align:center;width:30px}}
.ch{{background:#e8ecf0;font-weight:600;text-align:center;font-size:10px;color:#555}}
.rw th{{background:#1a1a2e;color:white;padding:5px 8px;font-size:11px;position:sticky;top:0}}
.rw td{{border:1px solid #e0e0e0;padding:3px 8px;font-size:11px}}
.vl{{font-weight:600;font-family:'Consolas',monospace;text-align:right}}
.img-row{{display:flex;gap:8px;overflow-x:auto;margin-bottom:20px;padding:8px;background:#fafafa;border-radius:8px}}
.img-row img{{max-height:360px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.1)}}
.img-label{{font-size:11px;color:#555;text-align:center;margin-top:4px}}
</style></head><body>
<h2>{base} — Table {int(tid.split('_')[1])+1}</h2>
<p>{nr}r x {nc}c, {nm} metrics{' (ESG)' if r.get('is_esg') else ''} | {r['tatr_cells']} TATR cells</p>
<div class='img-row'>
{segment_images}
</div>
<h3>Table Grid</h3>
<div class='tw'><table><tr><td class='ch'>#</td>{"".join(f'<td class="ch">C{ci}</td>' for ci in range(nc))}</tr>
{''.join(html_rows)}
</table></div>
<h3>Extracted Metrics</h3>
<div class='tw'><table class='rw'><thead><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Year</th><th>Scope</th><th>Type</th></tr></thead>
<tbody>{metrics_rows}</tbody></table></div>
</body></html>"""
        with open(os.path.join(page_dir, f"{tid}.html"), "w", encoding="utf-8") as f:
            f.write(html)

    idx_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:30px;background:#f0f2f4}}
h2{{color:#1a1a2e}}
.stats{{display:flex;gap:16px;margin-bottom:20px}}
.card{{background:white;border-radius:8px;padding:12px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.08)}}
.card .n{{font-size:24px;font-weight:700;color:#1a1a2e}}
.card .l{{font-size:11px;color:#888;text-transform:uppercase}}
.img-row{{display:flex;gap:8px;overflow-x:auto;margin-bottom:20px;padding:8px;background:#fafafa;border-radius:8px}}
.img-row img{{max-height:360px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.1)}}
.img-label{{font-size:11px;color:#555;text-align:center;margin-top:4px}}
.nav a{{display:inline-block;padding:10px 20px;margin:4px;background:#e8ecf0;border-radius:6px;text-decoration:none;color:#333;font-size:13px;font-weight:500}}
.nav a:hover{{background:#4a90d9;color:white}}
</style></head><body>
<h2>{base}</h2>
<div class='stats'>
<div class='card'><div class='n'>{n_tab}</div><div class='l'>Tables</div></div>
<div class='card'><div class='n'>{total_metrics}</div><div class='l'>Metrics</div></div>
<div class='card'><div class='n'>{n_esg}</div><div class='l'>ESG</div></div>
</div>
<div class='img-row'>
<div><img src="{base}_ocr.jpg"><div class="img-label">1. PaddleOCR</div></div>
<div><img src="{base}_layoutlmv3.jpg"><div class="img-label">2. LayoutLMv3 labels</div></div>
</div>
<div class='nav'>{" ".join(nav)}</div>
</body></html>"""
    with open(os.path.join(page_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(idx_html)
    with open(os.path.join(page_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _is_digit(t):
    return bool(t and t.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("-", "").replace("VND", "").strip().isdigit())


def _export_empty(page_dir, base):
    os.makedirs(page_dir, exist_ok=True)
    with open(os.path.join(page_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:30px;background:#f0f2f4}}
h2{{color:#1a1a2e}}
.img-row{{display:flex;gap:8px;overflow-x:auto;margin-bottom:20px;padding:8px;background:#fafafa;border-radius:8px}}
.img-row img{{max-height:360px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.1)}}
.img-label{{font-size:11px;color:#555;text-align:center;margin-top:4px}}
</style></head><body>
<h2>{base}</h2>
<p>No tables detected</p>
<div class='img-row'>
<div><img src="{base}_ocr.jpg"><div class="img-label">1. PaddleOCR</div></div>
<div><img src="{base}_layoutlmv3.jpg"><div class="img-label">2. LayoutLMv3 labels</div></div>
</div>
</body></html>""")
    with open(os.path.join(page_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump([], f)


def _export_all_json(all_results, out_dir):
    with open(os.path.join(out_dir, "all_tables.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

def _export_pending_classification(all_results, out_dir):
    pending = [
        m for r in all_results for m in r.get("extracted_metrics", [])
        if m.get("needs_classification")
    ]
    with open(os.path.join(out_dir, "pending_scope_classification.json"), "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(pending)} metrics can ClimateBERT resolve (pending_scope_classification.json)")
    return pending

def _export_master(out_dir):
    bases = sorted([e for e in os.listdir(out_dir)
                    if os.path.isdir(os.path.join(out_dir, e)) and not e.startswith("_")])
    total_tab = 0
    total_metrics = 0
    total_esg = 0
    hn = []
    for b in bases:
        jpath = os.path.join(out_dir, b, "tables.json")
        nt = 0
        if os.path.exists(jpath):
            with open(jpath) as f:
                data = json.load(f)
            nt = len(data)
            total_tab += nt
            total_metrics += sum(len(r.get("extracted_metrics", [])) for r in data)
            total_esg += sum(1 for r in data if r.get("is_esg"))
        hn.append(f'<a href="{b}/index.html">{b}<br><small>{nt} tables</small></a>')
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:30px;background:#f0f2f4}}
h2{{color:#1a1a2e}}
.stats{{display:flex;gap:16px;margin-bottom:20px}}
.card{{background:white;border-radius:8px;padding:12px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.08)}}
.card .n{{font-size:24px;font-weight:700;color:#1a1a2e}}
.card .l{{font-size:11px;color:#888;text-transform:uppercase}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.grid a{{display:block;background:white;border-radius:8px;padding:16px;text-decoration:none;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.08);text-align:center}}
.grid a:hover{{background:#e8f0fe;box-shadow:0 2px 8px rgba(0,0,0,0.12)}}
.grid a small{{color:#888;font-size:11px}}
</style></head><body>
<h2>Table Understanding — Results</h2>
<div class='stats'>
<div class='card'><div class='n'>{len(bases)}</div><div class='l'>Pages</div></div>
<div class='card'><div class='n'>{total_tab}</div><div class='l'>Tables</div></div>
<div class='card'><div class='n'>{total_metrics}</div><div class='l'>Metrics</div></div>
<div class='card'><div class='n'>{total_esg}</div><div class='l'>ESG</div></div>
</div>
<div class='grid'>{" ".join(hn)}</div>
</body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
