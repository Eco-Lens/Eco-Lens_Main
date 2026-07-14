"""
visualize_results.py
--------------------
Generate HTML report and scope-overlaid images from unified results.

Usage:
    python "4.SemanticMapping/visualize_results.py"
"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
from collections import Counter

UNIFIED_JSON = os.path.join("test", "output", "step4_semantic_mapping", "all_results_unified.json")
LAYOUT_JSON = os.path.join("test", "output", "step4_semantic_mapping", "0_layoutlmv3_layout.json")
TABLES_JSON = os.path.join("test", "output", "step3_table_understanding", "all_tables.json")
IMAGE_DIR = "test"
OUT_DIR = os.path.join("test", "output", "step5_visualization")

SCOPE_COLORS_HEX = {
    "Scope 1": "#ff4444",
    "Scope 2": "#ff8800",
    "Scope 3": "#44aaff",
    "Other": "#888888",
    "Mixed": "#aa44ff",
}

SCOPE_COLORS_RGB = {
    "Scope 1": (255, 68, 68),
    "Scope 2": (255, 136, 0),
    "Scope 3": (68, 170, 255),
    "Other": (136, 136, 136),
    "Mixed": (170, 68, 255),
}


def table_fingerprint(table):
    payload = {
        "page": table.get("page"),
        "table_id": table.get("table_id"),
        "bbox": table.get("bbox"),
        "segments": table.get("segments", []),
        "table_data": table.get("table_data", []),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_table_inheritance(unified_data, step3_tables):
    source = {
        (table["page"], table["table_id"]): table_fingerprint(table)
        for table in step3_tables
    }
    inherited = {
        (page, table["table_id"]): table.get("source_fingerprint")
        for page, data in unified_data.items()
        for table in data.get("tables", [])
    }
    if source != inherited:
        raise RuntimeError(
            "Step 4 table output is stale or differs from Step 3. "
            "Run run_scope_inference.py before visualization."
        )


def draw_scope_overlay(page_name, img_name, blocks_data, tables_data):
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font

    for bp in blocks_data:
        bbox = bp.get("bbox")
        if not bbox:
            continue
        scope = bp["scope"]
        color = SCOPE_COLORS_RGB.get(scope, (136, 136, 136))
        conf = bp.get("confidence", 0)
        val = bp.get("value")
        unit = bp.get("unit")
        x0 = int(bbox[0] * iw / 1000)
        y0 = int(bbox[1] * ih / 1000)
        x1 = int(bbox[2] * iw / 1000)
        y1 = int(bbox[3] * ih / 1000)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{scope} ({conf:.2f})"
        if val is not None and (isinstance(val, (int, float)) or val):
            label += f" | {val} {unit}" if unit else f" | {val}"
        ty = max(0, y0 - 20)
        tb = draw.textbbox((x0, ty), label, font=small_font)
        draw.rectangle(tb, fill=color)
        draw.text((x0, ty), label, fill=(255, 255, 255), font=small_font)

    # Draw tables with dashed scope color
    for t in tables_data:
        segments = t.get("segments") or [{
            "bbox": t.get("bbox") or t.get("layout_bbox")
        }]
        bboxes = [segment.get("bbox") for segment in segments if segment.get("bbox")]
        if not bboxes:
            continue
        scope = t.get("scope", "Other")
        color = SCOPE_COLORS_RGB.get(scope, (136, 136, 136))
        conf = t.get("confidence", 0)
        label = f"TABLE {t['table_id']}: {scope} ({conf:.2f})"
        for part_index, bbox_px in enumerate(bboxes):
            draw.rectangle(bbox_px, outline=color, width=5)
            part_label = label
            if len(bboxes) > 1:
                part_label += f" part {part_index + 1}"
            ty = max(0, bbox_px[1] - 22)
            tb = draw.textbbox((bbox_px[0], ty), part_label, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((bbox_px[0], ty), part_label, fill=(255, 255, 255), font=font)

    out_path = os.path.join(OUT_DIR, "overlay", f"{page_name}_scope.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path


def generate_html(unified_data, layout_data, scope_stats):
    # Count only non-Other blocks for display
    non_other_blocks = sum(
        1 for pd in unified_data.values()
        for b in pd.get("text_blocks", [])
        if b["scope"] != "Other"
    )
    total_tables = sum(len(pd["tables"]) for pd in unified_data.values())
    total_pages = len(unified_data)
    scope_bar = "".join(
        f'<span style="background:{SCOPE_COLORS_HEX[s]};padding:4px 12px;margin:2px;border-radius:4px;color:white;font-size:12px">{s}: {c}</span>'
        for s, c in sorted(scope_stats.items())
    )

    pages_html = []
    for page_name, pd in sorted(unified_data.items()):
        img_name = page_name + ".jpg"
        overlay_path = os.path.join("overlay", f"{page_name}_scope.jpg")

        blocks_rows = ""
        non_other = [b for b in pd.get("text_blocks", []) if b["scope"] != "Other"]
        for bi, b in enumerate(non_other):
            color = SCOPE_COLORS_HEX.get(b["scope"], "#888")
            val = b.get("value")
            unit = b.get("unit")
            val_str = f"{val} {unit}" if val is not None and unit is not None else ""
            blocks_rows += (
                f'<tr style="border-left:4px solid {color}">'
                f'<td>{bi}</td>'
                f'<td>{b["type"]}</td>'
                f'<td style="background:{color};color:white;font-weight:600">{b["scope"]}</td>'
                f'<td>{b["confidence"]:.3f}</td>'
                f'<td>{val_str}</td>'
                f'<td style="max-width:500px">{b["text"][:300]}</td>'
                f"</tr>\n"
            )

        tables_html = ""
        for t in pd.get("tables", []):
            tc = SCOPE_COLORS_HEX.get(t["scope"], "#888")
            # Build full table data
            td = t.get("table_data", [])
            ncols_full = max(len(r) for r in td) if td else 0
            data_preview = ""
            for ri, row in enumerate(td):
                cells_html = "".join(
                    "<td>" + (c if c else chr(8212)) + "</td>"
                    for c in row
                )
                data_preview += f"<tr><td class='rn'>{ri}</td>{cells_html}</tr>"

            # Nearby text
            nearby_html = ""
            for nb in t.get("nearby_text", []):
                nc = SCOPE_COLORS_HEX.get(nb["scope"], "#888")
                nearby_html += (
                    f'<div style="border-left:3px solid {nc};padding:4px 8px;margin:2px 0;font-size:11px">'
                    f'<span style="background:{nc};color:white;padding:1px 6px;border-radius:3px;font-size:10px">{nb["scope"]}</span> '
                    f'<span style="color:#888">{nb["text"][:150]}</span>'
                    f'</div>'
                )

            header_cells = "".join('<td class="ch">C' + str(ci) + '</td>' for ci in range(ncols_full))
            nearby_count = len(t.get('nearby_text', []))
            row_scopes = t.get('row_scopes', [])
            row_html = ""
            for rs in row_scopes:
                rc = SCOPE_COLORS_HEX.get(rs['scope'], '#888')
                row_html += '<div style="border-left:3px solid ' + rc + ';padding:3px 8px;margin:2px 0;font-size:11px">Row ' + str(rs['row']) + ': <span style="background:' + rc + ';color:white;padding:1px 6px;border-radius:3px;font-size:10px">' + rs['scope'] + '</span> ' + rs['label'][:80] + '</div>'
            tables_html += (
                f"<div class='card' style='border-left:5px solid {tc}'>"
                f"<h4>{t['table_id']} "
                f"<span style='background:{tc};color:white;padding:2px 10px;border-radius:4px;font-size:13px'>{t['scope']} ({t['confidence']:.2f})</span>"
                f" <span style='color:#888;font-size:12px'>{t['rows']}r x {t['cols']}c</span>"
                f"</h4>"
                f"<p style='font-size:11px;color:#888'>Source: {t.get('scope_source','')}</p>"
                f"<details open><summary style='cursor:pointer;font-weight:600;font-size:13px'>Full table data</summary>"
                f"<div class='tw'><table><tr><td class='ch'>#</td>{header_cells}</tr>"
                f"{data_preview}</table></div></details>"
                f"<details style='margin-top:8px'><summary style='cursor:pointer;font-weight:600;font-size:13px'>Nearby text ({nearby_count} blocks)</summary>"
                f"{nearby_html}</details>"
                f"<details style='margin-top:8px'><summary style='cursor:pointer;font-weight:600;font-size:13px'>Row scopes ({len(row_scopes)} classified)</summary>"
                f"{row_html}"
                f"</details>"
                f"</div>\n"
            )

        page_card = f"""
        <div class='page-card' id="{page_name}">
            <div class='page-header' onclick="togglePage(this)">
                <span class='page-title'>{page_name}</span>
                <span class='page-stats'>{len(non_other)} classified · {len(pd.get("tables",[]))} tables</span>
            </div>
            <div class='page-body'>
                <div class='img-row'>
                    <div><img src='{overlay_path}'><div class='img-label'>Scope overlay</div></div>
                </div>
                <h3>Text/Figure Blocks</h3>
                <table><thead><tr><th>#</th><th>Type</th><th>Scope</th><th>Confidence</th><th>Value</th><th>Text</th></tr></thead>
                <tbody>{blocks_rows}</tbody></table>
                <h3>Tables (classified from combined table content)</h3>
                {tables_html}
            </div>
        </div>
        """
        pages_html.append(page_card)

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
* {{box-sizing:border-box;margin:0;padding:0}}
body {{font-family:'Segoe UI',sans-serif;background:#f0f2f4;padding:20px}}
.header {{background:#1a1a2e;color:white;padding:20px 30px;border-radius:12px;margin-bottom:24px}}
.header h1 {{font-size:24px;font-weight:600}}
.header .sub {{font-size:13px;color:#aaa;margin-top:6px}}
.stats {{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
.card {{background:white;border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);flex:1;min-width:140px}}
.card .num {{font-size:28px;font-weight:700;color:#1a1a2e}}
.card .lbl {{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px}}
.scope-bar {{margin-bottom:24px}}
.page-card {{background:white;border-radius:10px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08);overflow:hidden}}
.page-header {{padding:14px 20px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;border-bottom:1px solid #eee;user-select:none}}
.page-header:hover {{background:#eef1f5}}
.page-title {{font-weight:600;font-size:15px;color:#1a1a2e}}
.page-stats {{font-size:12px;color:#888}}
.page-body {{padding:20px;display:block}}
.img-row {{display:flex;gap:8px;overflow-x:auto;margin-bottom:16px;padding:8px;background:#fafafa;border-radius:8px}}
.img-row img {{max-height:500px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.1);border:1px solid #ddd}}
.img-label {{font-size:11px;color:#555;text-align:center;margin-top:4px}}
table {{border-collapse:collapse;font-size:12px;width:100%;margin-bottom:16px}}
th {{background:#1a1a2e;color:white;padding:6px 10px;text-align:left;font-size:11px;position:sticky;top:0}}
td {{border:1px solid #e0e0e0;padding:5px 10px;vertical-align:top}}
tr:hover {{background:#f5f8ff}}
h3 {{margin:16px 0 8px;font-size:14px;color:#333}}
h4 {{margin:0 0 8px;font-size:13px;color:#555}}
.collapsed .page-body {{display:none}}
</style></head><body>
<div class='header'>
<h1>Eco-Lens — Scope Classification Results</h1>
<div class='sub'>{total_pages} pages · {non_other_blocks} classified blocks · {total_tables} tables</div>
</div>
<div class='stats'>
<div class='card'><div class='num'>{total_pages}</div><div class='lbl'>Pages</div></div>
<div class='card'><div class='num'>{non_other_blocks}</div><div class='lbl'>Classified Blocks</div></div>
<div class='card'><div class='num'>{total_tables}</div><div class='lbl'>Tables</div></div>
</div>
<div class='scope-bar'>{scope_bar}</div>
{"".join(pages_html)}
<script>
function togglePage(el) {{ el.parentElement.classList.toggle('collapsed'); }}
</script>
</body></html>"""

    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report: {out_path}")
    return out_path


def export_csv(unified_data):
    import csv
    rows = []
    for page_name, pd in sorted(unified_data.items()):
        for b in pd.get("text_blocks", []):
            rows.append({
                "page": page_name,
                "source": "block",
                "block_id": b.get("block_id", ""),
                "type": b.get("type", ""),
                "text": b.get("text", ""),
                "value": b.get("value", ""),
                "unit": b.get("unit", ""),
                "scope": b.get("scope", ""),
                "confidence": b.get("confidence", ""),
            })
        for t in pd.get("tables", []):
            rows.append({
                "page": page_name,
                "source": t["table_id"],
                "block_id": "",
                "type": "table",
                "text": f"Table {t['table_id']} ({t['rows']}r x {t['cols']}c)",
                "value": "",
                "unit": "",
                "scope": t.get("scope", ""),
                "confidence": t.get("confidence", ""),
            })
    csv_path = os.path.join(OUT_DIR, "scope_predictions_all.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "source", "block_id", "type", "text", "value", "unit", "scope", "confidence"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV export: {csv_path}")
    return csv_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading unified results...")
    with open(UNIFIED_JSON, "r", encoding="utf-8") as f:
        unified_data = json.load(f)

    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        step3_tables = json.load(f)
    validate_table_inheritance(unified_data, step3_tables)

    print("Loading layout JSON for bboxes...")
    with open(LAYOUT_JSON, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    # Build scope stats
    all_scopes = []
    for pd in unified_data.values():
        for b in pd.get("text_blocks", []):
            all_scopes.append(b["scope"])
        for t in pd.get("tables", []):
            all_scopes.append(t.get("scope", "Other"))
    scope_stats = dict(Counter(all_scopes))

    # Draw overlay images for ALL pages (including those without blocks/tables)
    print("Drawing scope overlays on images...")
    overlay_count = 0
    seen_pages = set()
    for page_name, pd in unified_data.items():
        seen_pages.add(page_name)
        img_name = page_name + ".jpg"
        out = draw_scope_overlay(
            page_name, img_name,
            pd.get("text_blocks", []),
            pd.get("tables", []),
        )
        if out:
            overlay_count += 1

    # Also output pages that exist in IMAGE_DIR but not in unified_data
    import glob
    for ext in [".jpg", ".jpeg", ".png"]:
        for img_path in glob.glob(os.path.join(IMAGE_DIR, "*" + ext)):
            base = os.path.splitext(os.path.basename(img_path))[0]
            if base not in seen_pages:
                out = draw_scope_overlay(base, os.path.basename(img_path), [], [])
                if out:
                    overlay_count += 1
                    seen_pages.add(base)

    print(f"  {overlay_count} overlay images saved ({len(seen_pages)} pages)")

    # Export HTML
    html_path = generate_html(unified_data, layout_data, scope_stats)

    # Export CSV
    csv_path = export_csv(unified_data)

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"  HTML:     file://{os.path.abspath(html_path)}")
    print(f"  CSV:      {csv_path}")


if __name__ == "__main__":
    main()
