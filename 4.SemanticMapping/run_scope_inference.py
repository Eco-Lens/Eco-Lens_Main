"""
run_scope_inference.py — ClimateBERT Scope classification.

Loads fine-tuned ClimateBERT, classifies text/figure blocks and tables.

Usage:
    python "4.SemanticMapping/run_scope_inference.py" \\
        --layout-json .../0_layoutlmv3_layout.json \\
        --tables-json .../all_tables.json \\
        --image-dir .../pages \\
        --out-dir .../step4_semantic_mapping
"""
import sys, os, json, time, re, hashlib, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline_core.config import CLIMATEBERT_TEMPERATURE, SCOPE_NAMES, SCHEMA_VERSION
from pipeline_core.utils import atomic_write_json, read_json_safe

MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_SCOPE_CONFIDENCE = 0.60

ALLOWED_UNIT_PATTERNS = [
    ("tCO2e", r"(?:t|tons?|tonnes?|metric\s+tons?)\s*(?:of\s*)?CO(?:2|₂|,)\s*(?:e|equivalent)?"),
    ("kgCO2e", r"kg\s*CO(?:2|₂)\s*(?:e|equivalent)?"),
    ("tCO2", r"t\s*CO(?:2|₂)"),
    ("ppm", r"ppm"),
    ("CO2e", r"CO(?:2|₂)\s*e"),
    ("mg/m3", r"mg\s*/\s*m(?:3|³)"),
    ("%", r"%|percent(?:age)?"),
    ("MWh", r"MWh"),
    ("kWh", r"kWh"),
    ("GWh", r"GWh"),
    ("GJ", r"GJ"),
]

ESG_SCOPE_CONTEXT = re.compile(
    r"\b(?:greenhouse\s+gas|ghg|emissions?|carbon|CO(?:2|₂)e?|electricity|"
    r"fuel|refrigerant|supply\s+chain|value\s+chain|net\s+zero|climate)\b",
    re.IGNORECASE,
)


def parse_number(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_value_unit(text):
    if not text:
        return None, None
    number_pattern = r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
    for unit, unit_pattern in ALLOWED_UNIT_PATTERNS:
        pattern = rf"(?<![\w.]){number_pattern}\s*(?:[-–—]\s*)?(?:{unit_pattern})(?![\w])"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = parse_number(match.group(1))
            if value is not None:
                return value, unit
    return None, None


def extract_explicit_scopes(text):
    scope_ids = {
        int(match.group(1))
        for match in re.finditer(r"\bscope\s*([123])\b", text or "", re.IGNORECASE)
    }
    return [f"Scope {scope_id}" for scope_id in sorted(scope_ids)]


def is_esg_eligible(text):
    """Semantic eligibility gate: True if text has ESG relevance."""
    if not text or len(text.strip()) < 3:
        return False
    if extract_explicit_scopes(text):
        return True
    if ESG_SCOPE_CONTEXT.search(text):
        return True
    value, unit = extract_value_unit(text)
    if value is not None and unit is not None:
        return True
    return False


def resolve_text_scope(text, prediction, value=None, unit=None):
    resolved = dict(prediction)
    model_scope = prediction["scope"]
    model_confidence = prediction["confidence"]
    explicit_scopes = extract_explicit_scopes(text)
    has_measurement = value is not None and unit is not None
    has_esg_context = bool(ESG_SCOPE_CONTEXT.search(text or ""))

    resolved.update({
        "model_scope": model_scope,
        "model_confidence": model_confidence,
        "explicit_scopes": explicit_scopes,
        "has_measurement": has_measurement,
    })

    if len(explicit_scopes) > 1:
        resolved.update({
            "scope": "Mixed", "scope_id": -1, "confidence": 1.0,
            "scope_source": "explicit_mentions",
            "decision_reason": "multiple_explicit_scopes",
            "scope_evidence": text[:300],
        })
        return resolved

    if len(explicit_scopes) == 1:
        explicit_scope = explicit_scopes[0]
        resolved.update({
            "scope": explicit_scope, "scope_id": int(explicit_scope[-1]),
            "confidence": 1.0, "scope_source": "explicit_mention",
            "decision_reason": "single_explicit_scope",
            "scope_evidence": text[:300],
        })
        return resolved

    if model_scope == "Other":
        resolved.update({
            "scope_source": "model", "decision_reason": "model_predicted_other",
            "scope_evidence": None,
        })
        return resolved

    # Non-Other model prediction: require ESG eligibility
    if not is_esg_eligible(text):
        resolved.update({
            "scope": "Other", "scope_id": 0,
            "confidence": round(prediction["probabilities"]["Other"], 4),
            "scope_source": "eligibility_gate",
            "decision_reason": "not_esg_eligible",
            "scope_evidence": None,
        })
        return resolved

    if model_confidence > TEXT_SCOPE_CONFIDENCE:
        resolved.update({
            "scope_source": (
                "model_with_measurement_and_esg_context"
                if has_measurement else "model_with_esg_context"
            ),
            "decision_reason": "supported_non_other_prediction",
            "scope_evidence": text[:300],
        })
        return resolved

    rejection_reasons = []
    if model_confidence <= TEXT_SCOPE_CONFIDENCE:
        rejection_reasons.append("low_confidence")
    if not has_esg_context:
        rejection_reasons.append("missing_esg_evidence")
    resolved.update({
        "scope": "Other", "scope_id": 0,
        "confidence": round(prediction["probabilities"]["Other"], 4),
        "scope_source": "policy_filter",
        "decision_reason": ",".join(rejection_reasons),
        "scope_evidence": None,
    })
    return resolved


def load_model(ckpt):
    print("Loading ClimateBERT Scope classifier...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded on {DEVICE}")
    return tokenizer, model


def predict_scope(text, tokenizer, model, temperature=CLIMATEBERT_TEMPERATURE):
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
        "scope": SCOPE_NAMES[pred_id],
        "scope_id": pred_id,
        "confidence": round(float(probs[pred_id]), 4),
        "probabilities": {SCOPE_NAMES[i]: round(float(probs[i]), 4) for i in range(len(probs))},
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


def get_image_size(page_name, image_dir):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(image_dir, page_name + ext)
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


def extract_table_content(table_data):
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
    _, _, combined = extract_table_content(table_data)
    if not combined or len(combined) < 5:
        return None
    prediction = predict_scope(combined, tokenizer, model)
    if prediction is None:
        return None
    value, unit = extract_value_unit(combined)
    return resolve_text_scope(combined, prediction, value, unit)


def classify_table_rows(table_data, tokenizer, model):
    if not table_data:
        return []
    results = []
    for ri, row in enumerate(table_data):
        if not row:
            continue
        label = row[0].strip()
        if not label or label.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit():
            continue
        context_parts = [label]
        for ci in range(1, min(3, len(row))):
            cell = row[ci].strip()
            if cell and not cell.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit():
                context_parts.append(cell)
        text = " ".join(context_parts)
        pred = predict_scope(text, tokenizer, model)
        if pred:
            value, unit = extract_value_unit(text)
            resolved = resolve_text_scope(text, pred, value, unit)
            results.append({
                "row": ri, "label": label,
                "scope": resolved["scope"], "confidence": resolved["confidence"],
                "scope_source": resolved["scope_source"],
            })
    return results


def match_tables_to_blocks(tables_data, block_preds, layout_data, tokenizer, model, image_dir):
    page_tables = {}
    for t in tables_data:
        page_tables.setdefault(t["page"], []).append(t)
    page_blocks = {}
    for bp in block_preds:
        p = _norm_page(bp["page"])
        page_blocks.setdefault(p, []).append(bp)
    table_results = []
    for page_name, tables in sorted(page_tables.items()):
        img_size = get_image_size(page_name, image_dir)
        for t in tables:
            content_pred = classify_table_via_content(t.get("table_data", []), tokenizer, model)
            row_scopes = classify_table_rows(t.get("table_data", []), tokenizer, model)
            supported_rows = [row for row in row_scopes if row["scope"] != "Other"]
            distinct_scopes = sorted({row["scope"] for row in supported_rows})
            if len(distinct_scopes) > 1:
                scope = "Mixed"
                confidence = min(row["confidence"] for row in supported_rows)
                source = "table_rows"
            elif len(distinct_scopes) == 1:
                scope = distinct_scopes[0]
                matching_rows = [row for row in supported_rows if row["scope"] == scope]
                confidence = sum(row["confidence"] for row in matching_rows) / len(matching_rows)
                source = "table_rows"
            else:
                scope = content_pred["scope"] if content_pred else "Other"
                confidence = content_pred["confidence"] if content_pred else 0.0
                source = "table_content" if content_pred else "unresolved"
            table_results.append({
                "page": page_name,
                "table_id": t["table_id"],
                "bbox": t.get("bbox"),
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
                "row_scopes": row_scopes,
            })
    return table_results


def main():
    ap = argparse.ArgumentParser(description="ClimateBERT Scope Classification")
    ap.add_argument("--layout-json", required=True, help="Layout blocks JSON from step 4a")
    ap.add_argument("--tables-json", required=True, help="Table data JSON from step 3")
    ap.add_argument("--image-dir", required=True, help="Page images directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--ckpt", default=None, help="ClimateBERT checkpoint path")
    ap.add_argument("--temperature", type=float, default=CLIMATEBERT_TEMPERATURE, help="Softmax temperature")
    ap.add_argument("--run-id", default=None, help="Run ID (for metadata)")
    args = ap.parse_args()

    t0 = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = args.ckpt or os.path.join(project_root, "Output", "4_Model_Classification_Scope", "checkpoint-2178")
    run_id = args.run_id or os.environ.get("RUN_ID", "unknown")

    tokenizer, model = load_model(ckpt)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nLoading layout JSON...")
    with open(args.layout_json, "r", encoding="utf-8") as f:
        layout_data = json.load(f)
    n_blocks = sum(len(p.get("blocks", [])) for p in layout_data.values())
    print(f"  {len(layout_data)} pages, {n_blocks} blocks")

    print("\nLoading tables JSON...")
    with open(args.tables_json, "r", encoding="utf-8") as f:
        tables_data = json.load(f)
    print(f"  {len(tables_data)} tables")

    # Classify text/figure blocks
    print("\nClassifying text/figure blocks...")
    block_preds = []
    skip_types = {"table", "table_text"}
    # toc/header/footer blocks: preserve in structure but don't send to ClimateBERT
    document_structure = {"toc": [], "header": [], "footer": [], "classified": []}

    for img_name, page in sorted(layout_data.items()):
        for bi, block in enumerate(page.get("blocks", [])):
            block_type = block.get("type", "")
            text = block.get("text", "").strip()

            # Preserve TOC, header, footer in document structure
            if block_type in ("toc", "header", "footer"):
                document_structure.setdefault(block_type, []).append({
                    "page": img_name, "block_id": bi, "type": block_type,
                    "text": text, "bbox": block.get("bbox"),
                })
                if block_type == "toc":
                    continue  # never send to ClimateBERT

            if block_type in skip_types:
                continue
            if not text or len(text) < 5:
                continue

            # Eligibility gate: skip non-ESG content before model call
            if not is_esg_eligible(text) and block_type in ("header", "footer"):
                continue

            model_pred = predict_scope(text, tokenizer, model, temperature=args.temperature)
            if model_pred is None:
                continue

            value, unit = extract_value_unit(text)
            pred = resolve_text_scope(text, model_pred, value, unit)

            entry = {
                "page": img_name, "block_id": bi, "type": block_type,
                "bbox": block.get("bbox"), "text": text,
                "value": value, "unit": unit,
                **pred,
            }
            block_preds.append(entry)
            document_structure["classified"].append(entry)

    classified_count = sum(1 for bp in block_preds if bp["scope"] != "Other")
    print(f"  {len(block_preds)} blocks classified ({classified_count} non-Other)")

    # Classify tables
    print("\nClassifying tables...")
    table_results = match_tables_to_blocks(tables_data, block_preds, layout_data, tokenizer, model, args.image_dir)
    print(f"  {len(table_results)} tables matched")

    # Stats
    block_scope_counts = Counter(bp["scope"] for bp in block_preds)
    print(f"\n  Block scope distribution: {dict(block_scope_counts)}")
    table_scope_counts = Counter(tr["scope"] for tr in table_results)
    print(f"  Table scope distribution: {dict(table_scope_counts)}")

    # Build unified output
    unified = {}
    for img_name in layout_data:
        p = _norm_page(img_name)
        unified.setdefault(p, {"page": p, "text_blocks": [], "tables": []})

    for bp in block_preds:
        p = _norm_page(bp["page"])
        unified[p]["text_blocks"].append({
            "block_id": bp["block_id"], "type": bp["type"],
            "bbox": bp.get("bbox"), "text": bp["text"],
            "value": bp.get("value"), "unit": bp.get("unit"),
            "scope": bp["scope"], "confidence": bp["confidence"],
            "scope_evidence": bp.get("scope_evidence"),
        })
    for tr in table_results:
        p = tr["page"]
        unified.setdefault(p, {"page": p, "text_blocks": [], "tables": []})
        unified[p]["tables"].append({
            "table_id": tr["table_id"], "bbox": tr.get("bbox"),
            "segments": tr.get("segments", []),
            "source_fingerprint": tr.get("source_fingerprint"),
            "table_data": tr["table_data"], "rows": tr["rows"], "cols": tr["cols"],
            "scope": tr["scope"], "scope_source": tr["scope_source"],
            "confidence": tr["confidence"],
            "nearby_text": tr.get("nearby_text", []),
            "row_scopes": tr.get("row_scopes", []),
        })

    # Write outputs atomically
    atomic_write_json(block_preds, os.path.join(args.out_dir, "scope_predictions.json"), schema_version=SCHEMA_VERSION)
    atomic_write_json(table_results, os.path.join(args.out_dir, "table_scope_predictions.json"), schema_version=SCHEMA_VERSION)
    atomic_write_json(unified, os.path.join(args.out_dir, "all_results_unified.json"), schema_version=SCHEMA_VERSION)

    # Also save document structure for downstream use
    atomic_write_json(document_structure, os.path.join(args.out_dir, "document_structure.json"), schema_version=SCHEMA_VERSION)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
