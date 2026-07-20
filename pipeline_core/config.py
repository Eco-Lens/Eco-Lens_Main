"""pipeline_core/config.py — Shared pipeline configuration."""

SCHEMA_VERSION = "1.0"

# Step definitions (single source of truth)
STEPS = [
    {"id": "convert",     "name": "PDF Rendering",      "icon": "\U0001f4c4", "weight": 8,  "desc": "Render PDF pages to JPG images"},
    {"id": "ocr",         "name": "OCR",                "icon": "\U0001f50d", "weight": 25, "desc": "Extract text with PaddleOCR"},
    {"id": "layout",      "name": "Layout Analysis",    "icon": "\U0001f4cd", "weight": 20, "desc": "Classify words via LayoutLMv3"},
    {"id": "tables",      "name": "Table Understanding", "icon": "\U0001f4ca", "weight": 18, "desc": "Extract tables via TATR"},
    {"id": "blocks",      "name": "Layout Blocks",      "icon": "\U0001f9f0",  "weight": 5,  "desc": "Group words into blocks"},
    {"id": "scope",       "name": "Scope Classification","icon": "\U0001f3af", "weight": 13, "desc": "Classify ESG scope via ClimateBERT"},
    {"id": "visualize",   "name": "Visualization",       "icon": "\U0001f4c8", "weight": 11, "desc": "Generate HTML, CSV, overlay images"},
]

STEP_WEIGHTS = {s["id"]: s["weight"] for s in STEPS}
TOTAL_WEIGHT = sum(s["weight"] for s in STEPS)

# LayoutLMv3 label mapping
LAYOUT_LABELS = {
    0: "O", 1: "chart", 2: "figure", 3: "footer", 4: "header",
    5: "ignore", 6: "table", 7: "table_text", 8: "text", 9: "toc",
}

# Default temperature for ClimateBERT
CLIMATEBERT_TEMPERATURE = 5.0

# Scope classification constants
SCOPE_NAMES = {0: "Other", 1: "Scope 1", 2: "Scope 2", 3: "Scope 3"}
