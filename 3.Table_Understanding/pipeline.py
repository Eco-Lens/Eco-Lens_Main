import sys, os, json, time, re, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw, ImageFont
import torch; import torchvision

from config import *
from utils import (
    merge_bboxes, bbox_area, expand_bbox,
    split_by_columns, split_by_vertical_gap,
    is_paragraph_cell, is_numeric, is_numeric_lenient, parse_number,
)
from tatr_engine import TATREngine


def run(ocr_path, labels_path, image_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    tatr = TATREngine()
    all_results = []

    for img_name in sorted(ocr_data.keys()):
        words = ocr_data.get(img_name, [])
        labels = label_data.get(img_name, [])
        if not words or not labels:
            continue

        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            continue

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
            {"text": words[i]["text"], "bbox": words[i]["bbox"]}
            for i in range(min(len(words), len(labels)))
            if labels[i] == TABLE_LABEL
        ]
        # Output 2: LayoutLMv3 label visualization
        _draw_layoutlmv3_vis(img, words, labels, page_dir, base)

        page_results = []
        page_log["n_table_tokens"] = len(table_tokens)

        if len(table_tokens) < MIN_TOKENS_PER_REGION:
            _export_empty(page_dir, base)
            print(f"  {img_name}: 0 tables (<{MIN_TOKENS_PER_REGION} table tokens)")
            all_results.extend(page_log["regions"])
            continue

        # Step A: Cluster table tokens vertically (separate stacked tables)
        vertical_clusters = split_by_vertical_gap(table_tokens, ih, VERTICAL_GAP_MULTIPLIER)

        # Step B: Within each vertical cluster, detect column gaps then build regions
        # Only table-labeled tokens (label == "table") are used — no all_tokens scan
        valid_regions = []
        for cluster in vertical_clusters:
            if len(cluster) < MIN_TOKENS_PER_REGION:
                page_log["skipped"].append({"reason": "too_few_table_tokens", "n": len(cluster)})
                continue

            col_groups = split_by_columns(cluster, iw, COLUMN_GAP_RATIO)

            for group in col_groups:
                if len(group) < MIN_TOKENS_PER_REGION:
                    continue

                bbox = merge_bboxes([t["bbox"] for t in group])
                crop_box = expand_bbox(bbox, EXPAND_MARGIN, EXPAND_MARGIN, iw, ih)

                # Reject oversized regions (likely leaked into text area)
                area_ratio = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]) / page_area
                if area_ratio > MAX_TABLE_REGION_AREA_RATIO:
                    page_log["skipped"].append({"reason": "region_too_large", "bbox": crop_box, "area_ratio": area_ratio})
                    continue

                # Check numeric content among table tokens only
                num_inside = sum(1 for t in group if is_numeric_lenient(t["text"]))
                if num_inside < 2:
                    page_log["skipped"].append({"reason": "no_numeric_content", "bbox": crop_box})
                    continue

                valid_regions.append({"bbox": bbox, "crop_box": crop_box, "tokens": group})

        page_log["n_regions"] = len(valid_regions)

        # Process each region
        for ti, reg in enumerate(valid_regions):
            crop_box = reg["crop_box"]
            crop_img = img.crop(crop_box)

            # Convert table-labeled tokens to crop coordinates
            # Only tokens with label == "table" are used for cell assignment
            crop_tokens = []
            for tok in reg["tokens"]:
                crop_tokens.append({
                    "text": tok["text"],
                    "bbox": [
                        max(0, tok["bbox"][0] - crop_box[0]),
                        max(0, tok["bbox"][1] - crop_box[1]),
                        min(crop_box[2] - crop_box[0], tok["bbox"][2] - crop_box[0]),
                        min(crop_box[3] - crop_box[1], tok["bbox"][3] - crop_box[1]),
                    ],
                })

            if len(crop_tokens) < MIN_TOKENS_PER_REGION:
                page_log["skipped"].append({"reason": "crop_too_few_tokens", "bbox": crop_box})
                continue

            result = _process_crop(
                tatr, crop_img, crop_tokens, reg["bbox"], crop_box,
                base, ti, page_dir, img
            )
            if result:
                page_results.append(result)
                page_log["regions"].append({
                    "table_id": f"table_{ti:02d}",
                    "bbox": reg["bbox"],
                    "crop_size": [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]],
                    "ocr_words_inside": len(crop_tokens),
                    "tatr_cells": result.get("tatr_cells", 0),
                    "output_shape": [result.get("rows", 0), result.get("cols", 0)],
                })
            else:
                page_log["skipped"].append({"reason": "processing_failed", "bbox": crop_box})

        # Draw all table regions on page overview
        _draw_page_overview(img, page_results, page_dir, base)

        # Write log
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


def _process_crop(tatr, crop_img, crop_tokens, raw_bbox, crop_box, base, ti, page_dir, full_img):
    if len(crop_tokens) < MIN_TOKENS_PER_REGION:
        return None

    t0 = time.time()
    cells = tatr.detect(crop_img)
    det_time = time.time() - t0
    n_rows = sum(1 for c in cells if c["class_name"] == "table row")
    n_cols = sum(1 for c in cells if c["class_name"] == "table column")

    if n_rows < MIN_ROWS or n_cols < MIN_COLS:
        return None

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
        txc = (tok["bbox"][0] + tok["bbox"][2]) / 2
        tyc = (tok["bbox"][1] + tok["bbox"][3]) / 2
        ri = _nearest(tyc, rows_sorted, "y")
        ci = _nearest(txc, cols_sorted, "x")
        if ri is not None and ci is not None:
            grid.setdefault((ri, ci), []).append(tok)

    nrg = len(rows_sorted)
    ncg = len(cols_sorted)
    table_grid = []
    for ri in range(nrg):
        row = []
        for ci in range(ncg):
            toks = grid.get((ri, ci), [])
            toks.sort(key=lambda t: t["bbox"][0])
            row.append(" ".join(t["text"] for t in toks).strip())
        table_grid.append(row)

    table_grid = _clean_table(table_grid)
    if len(table_grid) < MIN_ROWS or len(table_grid[0]) < MIN_COLS:
        return None

    extracted = _extract_esg(table_grid)
    has_data = any(is_numeric(cell) for row in table_grid for cell in row)
    has_esg = any(kw in cell.lower() for row in table_grid for cell in row for kw in ESG_KEYWORDS)
    if not has_data:
        return None

    _save_debug_images(full_img, crop_img, cells, crop_box, page_dir, ti, base)

    return {
        "page": base,
        "table_id": f"table_{ti:02d}",
        "bbox": raw_bbox,
        "crop_bbox": crop_box,
        "source_label": TABLE_LABEL,
        "table_data": table_grid,
        "rows": len(table_grid),
        "cols": len(table_grid[0]) if table_grid else 0,
        "extracted_metrics": extracted,
        "is_esg": has_esg,
        "tatr_cells": len(cells),
        "tatr_time": round(det_time, 2),
        "tokens_in_crop": len(crop_tokens),
    }


def _nearest(coord, items, axis="y"):
    best_i, best_d = None, float("inf")
    for i, it in enumerate(items):
        lo, hi = (it["bbox"][1], it["bbox"][3]) if axis == "y" else (it["bbox"][0], it["bbox"][2])
        if lo <= coord <= hi:
            return i
        d = min(abs(coord - lo), abs(coord - hi))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _clean_table(grid):
    if not grid:
        return []

    # Filter paragraph rows
    cleaned = []
    for row in grid:
        text_cells = [c for c in row if c.strip()]
        if len(text_cells) == 1 and not any(is_numeric_lenient(c) for c in row):
            text = row[0] if row else ""
            if is_paragraph_cell(text):
                continue
        cleaned.append(row)
    grid = cleaned

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


def _extract_esg(grid):
    metrics = []
    for ri, row in enumerate(grid):
        for ci, cell in enumerate(row):
            cell_text = cell.strip()
            if not cell_text:
                continue
            if not is_numeric_lenient(cell_text.replace("VND", "").strip()):
                continue
            metric_name = ""
            for ci2 in range(ci):
                candidate = row[ci2].strip()
                if candidate and not is_numeric_lenient(candidate.replace("VND", "").strip()):
                    metric_name = candidate
                    break
            year = None
            ym = re.findall(YEAR_PATTERN, cell_text)
            if ym:
                year = int(ym[-1][1])
            unit = ""
            for p in ESG_UNIT_PATTERNS:
                m = re.search(p, cell_text, re.I)
                if m:
                    unit = m.group(0)
                    break
            if not unit and metric_name:
                for p in ESG_UNIT_PATTERNS:
                    m = re.search(p, metric_name, re.I)
                    if m:
                        unit = m.group(0)
                        break
            scope = ""
            for kw in ["scope 1", "scope 2", "scope 3", "scope i", "scope ii", "scope iii"]:
                if kw in metric_name.lower():
                    parts = kw.split()
                    scope = f"Scope {parts[-1].upper()}"
                    break
            value = parse_number(cell_text)
            is_esg_metric = any(kw in (metric_name + " " + cell_text).lower() for kw in ESG_KEYWORDS)
            if value is not None:
                metrics.append({
                    "metric": metric_name, "value": value, "unit": unit,
                    "year": year, "scope": scope, "is_esg": is_esg_metric,
                    "row": ri, "col": ci,
                })
    return metrics


def _draw_ocr_vis(img, words, page_dir, base):
    draw = ImageDraw.Draw(img)
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
    img.save(os.path.join(page_dir, f"{base}_ocr.jpg"))


_LABEL_COLORS = {
    "O": (128, 128, 128), "chart": (255, 165, 0), "figure": (128, 0, 128),
    "footer": (165, 42, 42), "header": (0, 102, 255), "ignore": (180, 180, 180),
    "table": (255, 0, 0), "table_text": (255, 0, 255),
    "text": (0, 180, 0), "toc": (0, 200, 200),
}


def _draw_layoutlmv3_vis(img, words, labels, page_dir, base):
    draw = ImageDraw.Draw(img)
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
    img.save(os.path.join(page_dir, f"{base}_layoutlmv3.jpg"))


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
        cb = r["crop_bbox"]
        draw.rectangle(cb, outline=col, width=3)
        label = f"table_{ti:02d}"
        draw.text((cb[0] + 4, max(0, cb[1] - 20)), label, fill=col, font=font)
    vis.save(os.path.join(page_dir, f"{base}_page_with_table_bboxes.jpg"))


def _save_debug_images(full_img, crop_img, tatr_cells, crop_box, page_dir, ti, base):
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        font = ImageFont.load_default()

    # Crop image
    crop_img.save(os.path.join(page_dir, f"table_{ti:02d}_crop.jpg"))

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
    vis2.save(os.path.join(page_dir, f"table_{ti:02d}_tatr_overlay.jpg"))


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
                f"<tr><td>{m['metric']}</td><td class='vl'>{m['value']}</td>"
                f"<td>{m['unit']}</td><td>{m['year'] or ''}</td>"
                f"<td>{m['scope']}</td><td>{'ESG' if m['is_esg'] else ''}</td></tr>"
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
<div><img src='table_{int(tid.split('_')[1]):02d}_crop.jpg'><div class="img-label">3. Table crop</div></div>
<div><img src='table_{int(tid.split('_')[1]):02d}_tatr_overlay.jpg'><div class="img-label">4. TATR structure</div></div>
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
