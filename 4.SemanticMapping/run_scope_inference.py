"""
run_scope_inference.py
---------------------
Load fine-tuned ClimateBERT Scope classifier.
Classify text/figure blocks, then assign scope to tables by:
  1. Classifying table content (headers + row labels) directly
  2. Checking nearby classified text blocks for context
  3. Combining both signals for robust table scope

Usage:
    python "4.SemanticMapping/run_scope_inference.py"
"""
import sys, os, json, time, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = os.path.join("Output", "4_Model_Classification_Scope", "checkpoint-2178")
LAYOUT_JSON = os.path.join("test", "output", "step4_semantic_mapping", "0_layoutlmv3_layout.json")
TABLES_JSON = os.path.join("test", "output", "step3_table_understanding", "all_tables.json")
IMAGE_DIR = "test"
OUT_DIR = os.path.join("test", "output", "step4_semantic_mapping")

MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 5.0
ID2LABEL = {0: "Other", 1: "Scope 1", 2: "Scope 2", 3: "Scope 3"}
Y_PROXIMITY = 150

TEXT_SCOPE_CONFIDENCE = 0.60
ALLOWED_UNIT_PATTERNS = [
    ("tCO2", r"t\s*CO(?:2|₂)"),
    ("ppm", r"ppm"),
    ("CO2e", r"CO(?:2|₂)\s*e"),
    ("mg/m3", r"mg\s*/\s*m(?:3|³)"),
]


def parse_number(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def _is_year(num):
    """Heuristic: 4-digit number between 1900-2099 is likely a year."""
    return 1900 <= num <= 2099


def extract_value_unit(text):
    if not text:
        return None, None
    number_pattern = r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
    for unit, unit_pattern in ALLOWED_UNIT_PATTERNS:
        pattern = rf"(?<![\w.]){number_pattern}\s*(?:[-–—]\s*)?(?:{unit_pattern})(?![\w])"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = parse_number(match.group(1))
        if value is not None and value > 0 and not _is_year(value):
            return value, unit
    return None, None


def load_model():
    print("Loading ClimateBERT Scope classifier...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT)
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded on {DEVICE}")
    return tokenizer, model


def predict_scope(text, tokenizer, model, temperature=TEMPERATURE):
    if not text or len(text.strip()) < 3:
        return None
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits / temperature
        probs = F.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    return {
        "scope": ID2LABEL[pred_id],
        "scope_id": pred_id,
        "confidence": round(float(probs[pred_id]), 4),
        "probabilities": {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))},
    }


def _norm_page(name):
    base, _ = os.path.splitext(name)
    return base


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


def get_image_size(page_name):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(IMAGE_DIR, page_name + ext)
        if os.path.exists(path):
            with Image.open(path) as img:
                return img.size
    return None


def normalize_bbox(bbox_px, w, h):
    return [
        round(1000 * bbox_px[0] / w),
        round(1000 * bbox_px[1] / h),
        round(1000 * bbox_px[2] / w),
        round(1000 * bbox_px[3] / h),
    ]


def bbox_center_y(bbox):
    return (bbox[1] + bbox[3]) / 2


def bbox_x_overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def extract_table_content(table_data):
    """
    Extract textual content from a table for scope classification.
    Returns: headers text, row labels text, combined text.
    """
    if not table_data:
        return "", "", ""

    cells = []
    for row in table_data:
        for cell in row:
            text = re.sub(r"\s+", " ", str(cell)).strip()
            if text:
                cells.append(text)
    combined = " ".join(cells)
    return "", "", combined


def classify_table_via_content(table_data, tokenizer, model):
    """
    Classify table by its headers, row labels, and optional section title.
    section_title: text from the nearest block above the table (table heading).
    """
    _, _, combined = extract_table_content(table_data)
    if not combined or len(combined) < 5:
        return None
    return predict_scope(combined, tokenizer, model)


def classify_table_rows(table_data, tokenizer, model):
    """
    Classify each row of a table individually based on its first-column label.
    Returns list of {row_index, label, scope, confidence}.
    """
    if not table_data:
        return []
    results = []
    for ri, row in enumerate(table_data):
        if not row:
            continue
        label = row[0].strip()
        if not label or label.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit():
            continue
        # Use row label + first non-empty text cell for context
        context_parts = [label]
        for ci in range(1, min(3, len(row))):
            cell = row[ci].strip()
            if cell and not cell.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit():
                context_parts.append(cell)
        text = " ".join(context_parts)
        pred = predict_scope(text, tokenizer, model)
        if pred:
            results.append({"row": ri, "label": label, "scope": pred["scope"], "confidence": pred["confidence"]})
    return results


def match_tables_to_blocks(tables_data, block_preds, layout_data, tokenizer, model):
    """
    Assign scope to each table using:
      1. Table content classification (headers + row labels → ClimateBERT)
      2. Per-row scope classification (each row's label individually)
      3. Nearby text blocks' scope (majority vote)
    Table output includes both overall scope and per-row breakdown.
    """
    page_tables = {}
    for t in tables_data:
        page_tables.setdefault(t["page"], []).append(t)

    page_blocks = {}
    for bp in block_preds:
        p = _norm_page(bp["page"])
        page_blocks.setdefault(p, []).append(bp)

    table_results = []

    for page_name, tables in sorted(page_tables.items()):
        img_size = get_image_size(page_name)
        blocks = page_blocks.get(page_name, [])

        for t in tables:
            bbox_px = t.get("bbox") or t.get("layout_bbox")
            bbox_norm = normalize_bbox(bbox_px, *img_size) if img_size else None
            content_pred = classify_table_via_content(t.get("table_data", []), tokenizer, model)
            scope = content_pred["scope"] if content_pred else "Other"
            confidence = content_pred["confidence"] if content_pred else 0.0
            source = "table_content" if content_pred else "unresolved"

            table_results.append({
                "page": page_name,
                "table_id": t["table_id"],
                "bbox": bbox_px,
                "layout_bbox": t.get("layout_bbox"),
                "crop_bbox": t.get("crop_bbox"),
                "segments": t.get("segments", []),
                "source_fingerprint": table_fingerprint(t),
                "table_data": t["table_data"],
                "rows": t["rows"],
                "cols": t["cols"],
                "scope": scope,
                "confidence": round(confidence, 3),
                "scope_source": source,
                "section_title": None,
                "nearby_text": [],
                "row_scopes": [],
            })

    return table_results


def main():
    t0 = time.time()
    tokenizer, model = load_model()

    print("\nLoading layout JSON...")
    with open(LAYOUT_JSON, "r", encoding="utf-8") as f:
        layout_data = json.load(f)
    n_blocks = sum(len(p.get("blocks", [])) for p in layout_data.values())
    print(f"  {len(layout_data)} pages, {n_blocks} blocks")

    print("\nLoading all_tables.json...")
    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        tables_data = json.load(f)
    print(f"  {len(tables_data)} tables")

    # Classify text/figure blocks (skip table/table_text — they're handled by step 3 → step 4b)
    print("\nClassifying text/figure blocks...")
    block_preds = []
    skip_types = {"table", "table_text"}
    for img_name, page in sorted(layout_data.items()):
        for bi, block in enumerate(page.get("blocks", [])):
            if block.get("type") in skip_types:
                continue
            text = block.get("text", "").strip()
            if not text or len(text) < 5:
                continue
            pred = predict_scope(text, tokenizer, model)
            if pred is None:
                continue

            value, unit = extract_value_unit(text)

            # Text/figure scope requires a high-confidence supported measurement.
            if pred["scope"] != "Other":
                has_value = isinstance(value, (int, float)) and value > 0
                has_allowed_unit = unit in {name for name, _ in ALLOWED_UNIT_PATTERNS}
                if pred["confidence"] <= TEXT_SCOPE_CONFIDENCE or not has_value or not has_allowed_unit:
                    pred["scope"] = "Other"
                    pred["scope_id"] = 0
                    pred["confidence"] = round(pred["probabilities"]["Other"], 4)

            block_preds.append({
                "page": img_name,
                "block_id": bi,
                "type": block.get("type", "unknown"),
                "bbox": block.get("bbox"),
                "text": text,
                "value": value,
                "unit": unit,
                **pred,
            })
    print(f"  {len(block_preds)} blocks classified")

    # Classify each table from its combined cell content.
    print("\nClassifying tables from combined table content...")
    table_results = match_tables_to_blocks(tables_data, block_preds, layout_data, tokenizer, model)
    print(f"  {len(table_results)} tables matched")

    # Stats
    block_scope_counts = Counter(bp["scope"] for bp in block_preds)
    print(f"\n  Block scope distribution: {dict(block_scope_counts)}")
    table_scope_counts = Counter(tr["scope"] for tr in table_results)
    print(f"  Table scope distribution: {dict(table_scope_counts)}")
    table_source_counts = Counter(tr["scope_source"] for tr in table_results)
    print(f"  Table scope source: {dict(table_source_counts)}")

    # Sample confidence values
    block_confs = [bp["confidence"] for bp in block_preds]
    if block_confs:
        print(f"  Block confidence range: {min(block_confs):.3f} - {max(block_confs):.3f} (avg: {np.mean(block_confs):.3f})")

    # Build unified output — include ALL pages from layout JSON
    unified = {}
    for img_name in layout_data:
        p = _norm_page(img_name)
        unified.setdefault(p, {"page": p, "text_blocks": [], "tables": []})

    for bp in block_preds:
        p = _norm_page(bp["page"])
        unified[p]["text_blocks"].append({
            "block_id": bp["block_id"],
            "type": bp["type"],
            "bbox": bp["bbox"],
            "text": bp["text"],
            "value": bp.get("value"),
            "unit": bp.get("unit"),
            "scope": bp["scope"],
            "confidence": bp["confidence"],
        })

    for tr in table_results:
        p = tr["page"]
        unified.setdefault(p, {"page": p, "text_blocks": [], "tables": []})
        unified[p]["tables"].append({
            "table_id": tr["table_id"],
            "bbox": tr["bbox"],
            "layout_bbox": tr.get("layout_bbox"),
            "crop_bbox": tr.get("crop_bbox"),
            "segments": tr.get("segments", []),
            "source_fingerprint": tr.get("source_fingerprint"),
            "table_data": tr["table_data"],
            "rows": tr["rows"],
            "cols": tr["cols"],
            "scope": tr["scope"],
            "scope_source": tr["scope_source"],
            "confidence": tr["confidence"],
            "nearby_text": tr.get("nearby_text", []),
            "row_scopes": tr.get("row_scopes", []),
        })

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)

    block_out = os.path.join(OUT_DIR, "scope_predictions.json")
    with open(block_out, "w", encoding="utf-8") as f:
        json.dump(block_preds, f, ensure_ascii=False, indent=2)
    print(f"\nSaved block predictions to {block_out}")

    tables_out = os.path.join(OUT_DIR, "table_scope_predictions.json")
    with open(tables_out, "w", encoding="utf-8") as f:
        json.dump(table_results, f, ensure_ascii=False, indent=2)
    print(f"Saved table predictions to {tables_out}")

    unified_out = os.path.join(OUT_DIR, "all_results_unified.json")
    with open(unified_out, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)
    print(f"Saved unified output to {unified_out}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
