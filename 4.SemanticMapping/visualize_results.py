"""
visualize_results.py — Generate HTML report, CSV export, and scope-overlaid images.

Usage:
    python "4.SemanticMapping/visualize_results.py" \\
        --unified-json .../all_results_unified.json \\
        --layout-json .../0_layoutlmv3_layout.json \\
        --tables-json .../all_tables.json \\
        --image-dir .../pages \\
        --out-dir .../step5_visualization
"""
import sys, os, json, hashlib, csv, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
from collections import Counter

from pipeline_core.config import SCHEMA_VERSION
from pipeline_core.utils import atomic_write_json

SCOPE_COLORS_HEX = {
    "Scope 1": "#ff4444", "Scope 2": "#ff8800", "Scope 3": "#44aaff",
    "Other": "#888888", "Mixed": "#aa44ff",
}
SCOPE_COLORS_RGB = {
    "Scope 1": (255, 68, 68), "Scope 2": (255, 136, 0),
    "Scope 3": (68, 170, 255), "Other": (136, 136, 136), "Mixed": (170, 68, 255),
}


def table_fingerprint(table):
    payload = {
        "page": table.get("page"), "table_id": table.get("table_id"),
        "bbox": table.get("bbox"), "segments": table.get("segments", []),
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


def draw_scope_overlay(page_name, img_path, blocks_data, tables_data, out_dir):
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for bp in blocks_data:
        bbox = bp.get("bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        scope = bp.get("scope", "Other")
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

    for t in tables_data:
        segments = t.get("segments") or [{
            "bbox": t.get("bbox") or t.get("layout_bbox")
        }]
        bboxes = []
        for seg in segments:
            sb = seg.get("bbox")
            if sb and isinstance(sb, (list, tuple)) and len(sb) >= 4:
                bboxes.append(sb)
        if not bboxes:
            continue
        scope = t.get("scope", "Other")
        color = SCOPE_COLORS_RGB.get(scope, (136, 136, 136))
        conf = t.get("confidence", 0)
        tid = t.get("table_id", "?")
        label = f"TABLE {tid}: {scope} ({conf:.2f})"
        for part_index, bbox_px in enumerate(bboxes):
            draw.rectangle(bbox_px, outline=color, width=5)
            part_label = label
            if len(bboxes) > 1:
                part_label += f" part {part_index + 1}"
            ty = max(0, bbox_px[1] - 22)
            tb = draw.textbbox((bbox_px[0], ty), part_label, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((bbox_px[0], ty), part_label, fill=(255, 255, 255), font=font)

    overlay_dir = os.path.join(out_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    out_path = os.path.join(overlay_dir, f"{page_name}_scope.jpg")
    img.save(out_path)
    return out_path


def generate_html(unified_data, layout_data, scope_stats):
    non_other_blocks = sum(
        1 for pd in unified_data.values()
        for b in pd.get("text_blocks", [])
        if b.get("scope", "Other") != "Other"
    )
    total_tables = sum(len(pd.get("tables", [])) for pd in unified_data.values())
    total_pages = len(unified_data)
    scope_bar = "".join(
        f'<span style="background:{SCOPE_COLORS_HEX[s]};padding:4px 12px;margin:2px;border-radius:4px;color:white;font-size:12px">{s}: {c}</span>'
        for s, c in sorted(scope_stats.items())
    )
    pages_html = []
    for page_name, pd in sorted(unified_data.items()):
        overlay_path = os.path.join("overlay", f"{page_name}_scope.jpg")
        blocks_rows = ""
        non_other = [b for b in pd.get("text_blocks", []) if b.get("scope", "Other") != "Other"]
        for bi, b in enumerate(non_other):
            bscope = b.get("scope", "Other")
            color = SCOPE_COLORS_HEX.get(bscope, "#888")
            val = b.get("value")
            unit = b.get("unit")
            val_str = f"{val} {unit}" if val is not None and unit is not None else ""
            btype = b.get("type", "unknown")
            bconf = b.get("confidence", 0)
            btext = (b.get("text", "") or "")[:300]
            blocks_rows += (
                f'<tr style="border-left:4px solid {color}">'
                f'<td>{bi}</td><td>{btype}</td>'
                f'<td style="background:{color};color:white;font-weight:600">{bscope}</td>'
                f'<td>{bconf:.3f}</td><td>{val_str}</td>'
                f'<td style="max-width:500px">{btext}</td></tr>\n'
            )
        tables_html = ""
        for t in pd.get("tables", []):
            tc = SCOPE_COLORS_HEX.get(t.get("scope", "Other"), "#888")
            td = t.get("table_data", [])
            ncols_full = max(len(r) for r in td) if td else 0
            data_preview = ""
            for ri, row in enumerate(td):
                cells_html = "".join(
                    "<td>" + (c if c else chr(8212)) + "</td>" for c in row
                )
                data_preview += f"<tr><td class='rn'>{ri}</td>{cells_html}</tr>"
            nearby_html = ""
            for nb in t.get("nearby_text", []):
                nc = SCOPE_COLORS_HEX.get(nb.get("scope", "Other"), "#888")
                nearby_html += (
                    f'<div style="border-left:3px solid {nc};padding:4px 8px;margin:2px 0;font-size:11px">'
                    f'<span style="background:{nc};color:white;padding:1px 6px;border-radius:3px;font-size:10px">{nb.get("scope","")}</span> '
                    f'<span style="color:#888">{str(nb.get("text",""))[:150]}</span></div>'
                )
            header_cells = "".join(f'<td class="ch">C{ci}</td>' for ci in range(ncols_full))
            row_scopes = t.get("row_scopes", [])
            row_html = ""
            for rs in row_scopes:
                rc = SCOPE_COLORS_HEX.get(rs.get("scope", "Other"), "#888")
                row_html += f'<div style="border-left:3px solid {rc};padding:3px 8px;margin:2px 0;font-size:11px">Row {rs.get("row","")}: <span style="background:{rc};color:white;padding:1px 6px;border-radius:3px;font-size:10px">{rs.get("scope","")}</span> {str(rs.get("label",""))[:80]}</div>'
            tscope = t.get("scope", "Other")
        tables_html += (
                f"<div class='card' style='border-left:5px solid {tc}'>"
                f"<h4>{t.get('table_id','')} "
                f"<span style='background:{tc};color:white;padding:2px 10px;border-radius:4px;font-size:13px'>{tscope} ({t.get('confidence',0):.2f})</span>"
                f" <span style='color:#888;font-size:12px'>{t.get('rows','?')}r x {t.get('cols','?')}c</span></h4>"
                f"<p style='font-size:11px;color:#888'>Source: {t.get('scope_source','')}</p>"
                f"<details open><summary style='cursor:pointer;font-weight:600;font-size:13px'>Full table data</summary>"
                f"<div class='tw'><table><tr><td class='ch'>#</td>{header_cells}</tr>"
                f"{data_preview}</table></div></details>"
                f"<details style='margin-top:8px'><summary style='cursor:pointer;font-weight:600;font-size:13px'>Nearby text ({len(t.get('nearby_text',[]))} blocks)</summary>"
                f"{nearby_html}</details>"
                f"<details style='margin-top:8px'><summary style='cursor:pointer;font-weight:600;font-size:13px'>Row scopes ({len(row_scopes)} classified)</summary>"
                f"{row_html}</details></div>\n"
            )
        page_card = f"""
        <div class='page-card' id="{page_name}">
            <div class='page-header' onclick="togglePage(this)">
                <span class='page-title'>{page_name}</span>
                <span class='page-stats'>{len(non_other)} classified · {len(pd.get("tables",[]))} tables</span>
            </div>
            <div class='page-body'>
                <div class='img-row'><div><img src='{overlay_path}'><div class='img-label'>Scope overlay</div></div></div>
                <h3>Text/Figure Blocks</h3>
                <table><thead><tr><th>#</th><th>Type</th><th>Scope</th><th>Confidence</th><th>Value</th><th>Text</th></tr></thead>
                <tbody>{blocks_rows}</tbody></table>
                <h3>Tables</h3>
                {tables_html}
            </div>
        </div>"""
        pages_html.append(page_card)

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f0f2f4;padding:20px}}
.header{{background:#1a1a2e;color:white;padding:20px 30px;border-radius:12px;margin-bottom:24px}}
.header h1{{font-size:24px;font-weight:600}}
.header .sub{{font-size:13px;color:#aaa;margin-top:6px}}
.stats{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
.card{{background:white;border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);flex:1;min-width:140px}}
.card .num{{font-size:28px;font-weight:700;color:#1a1a2e}}
.card .lbl{{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px}}
.scope-bar{{margin-bottom:24px}}
.page-card{{background:white;border-radius:10px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08);overflow:hidden}}
.page-header{{padding:14px 20px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;border-bottom:1px solid #eee;user-select:none}}
.page-header:hover{{background:#eef1f5}}
.page-title{{font-weight:600;font-size:15px;color:#1a1a2e}}
.page-stats{{font-size:12px;color:#888}}
.page-body{{padding:20px;display:block}}
.img-row{{display:flex;gap:8px;overflow-x:auto;margin-bottom:16px;padding:8px;background:#fafafa;border-radius:8px}}
.img-row img{{max-height:500px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.1);border:1px solid #ddd}}
.img-label{{font-size:11px;color:#555;text-align:center;margin-top:4px}}
table{{border-collapse:collapse;font-size:12px;width:100%;margin-bottom:16px}}
th{{background:#1a1a2e;color:white;padding:6px 10px;text-align:left;font-size:11px;position:sticky;top:0}}
td{{border:1px solid #e0e0e0;padding:5px 10px;vertical-align:top}}
tr:hover{{background:#f5f8ff}}
h3{{margin:16px 0 8px;font-size:14px;color:#333}}
h4{{margin:0 0 8px;font-size:13px;color:#555}}
.collapsed .page-body{{display:none}}
</style></head><body>
<div class='header'><h1>Eco-Lens — Scope Classification Results</h1><div class='sub'>{total_pages} pages · {non_other_blocks} classified blocks · {total_tables} tables</div></div>
<div class='stats'><div class='card'><div class='num'>{total_pages}</div><div class='lbl'>Pages</div></div><div class='card'><div class='num'>{non_other_blocks}</div><div class='lbl'>Classified Blocks</div></div><div class='card'><div class='num'>{total_tables}</div><div class='lbl'>Tables</div></div></div>
<div class='scope-bar'>{scope_bar}</div>
{"".join(pages_html)}
<script>function togglePage(el){{el.parentElement.classList.toggle('collapsed');}}</script>
</body></html>"""
    out_path = os.path.join(args.out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report: {out_path}")
    return out_path


def export_csv(unified_data, out_path):
    rows = []
    for page_name, pd in sorted(unified_data.items()):
        for b in pd.get("text_blocks", []):
            rows.append({
                "page": page_name, "source": "block",
                "block_id": b.get("block_id", ""), "type": b.get("type", ""),
                "text": b.get("text", ""), "value": b.get("value", ""),
                "unit": b.get("unit", ""), "scope": b.get("scope", ""),
                "confidence": b.get("confidence", ""),
            })
        for t in pd.get("tables", []):
            rows.append({
                "page": page_name, "source": t.get("table_id", ""),
                "block_id": "", "type": "table",
                "text": f"Table {t.get('table_id','')} ({t.get('rows','?')}r x {t.get('cols','?')}c)",
                "value": "", "unit": "", "scope": t.get("scope", ""),
                "confidence": t.get("confidence", ""),
            })
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "source", "block_id", "type", "text", "value", "unit", "scope", "confidence"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV export: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Visualization — HTML, CSV, overlay images")
    ap.add_argument("--unified-json", required=True, help="Unified results JSON from step 4b")
    ap.add_argument("--layout-json", required=True, help="Layout blocks JSON from step 4a")
    ap.add_argument("--tables-json", required=True, help="Table data JSON from step 3")
    ap.add_argument("--image-dir", required=True, help="Page images directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--run-id", default=None, help="Run ID (for metadata)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading unified results...")
    with open(args.unified_json, "r", encoding="utf-8") as f:
        unified_data = json.load(f)
    with open(args.tables_json, "r", encoding="utf-8") as f:
        step3_tables = json.load(f)
    validate_table_inheritance(unified_data, step3_tables)

    with open(args.layout_json, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    # Scope stats
    all_scopes = []
    for pd in unified_data.values():
        for b in pd.get("text_blocks", []):
            all_scopes.append(b.get("scope", "Other"))
        for t in pd.get("tables", []):
            all_scopes.append(t.get("scope", "Other"))
    scope_stats = dict(Counter(all_scopes))

    # Draw overlays
    print("Drawing scope overlays...")
    overlay_count = 0
    seen_pages = set()
    for page_name, pd in unified_data.items():
        try:
            seen_pages.add(page_name)
            candidates = [os.path.join(args.image_dir, page_name + ext) for ext in [".jpg", ".jpeg", ".png"]]
            img_path = next((p for p in candidates if os.path.exists(p)), None)
            if not img_path:
                print(f"  Skipping {page_name}: image not found")
                continue
            out = draw_scope_overlay(page_name, img_path, pd.get("text_blocks", []), pd.get("tables", []), args.out_dir)
            if out:
                overlay_count += 1
        except Exception as e:
            print(f"  Error drawing overlay for {page_name}: {e}")
            continue

    import glob
    for ext in [".jpg", ".jpeg", ".png"]:
        for img_path in glob.glob(os.path.join(args.image_dir, "*" + ext)):
            base = os.path.splitext(os.path.basename(img_path))[0]
            if base not in seen_pages and os.path.exists(img_path):
                try:
                    out = draw_scope_overlay(base, img_path, [], [], args.out_dir)
                    if out:
                        overlay_count += 1
                        seen_pages.add(base)
                except Exception as e:
                    print(f"  Error drawing overlay for {base}: {e}")
                    continue

    print(f"  {overlay_count} overlay images")

    # Generate HTML + CSV
    try:
        generate_html(unified_data, layout_data, scope_stats)
    except Exception as e:
        print(f"  Error generating HTML: {e}")

    csv_path = os.path.join(args.out_dir, "scope_predictions_all.csv")
    try:
        export_csv(unified_data, csv_path)
    except Exception as e:
        print(f"  Error exporting CSV: {e}")

    print(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
