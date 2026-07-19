"""pipeline_core/context.py — RunContext: per-run directory isolation."""

import os, time, json, shutil, glob
from pathlib import Path

SCHEMA_VERSION = "1.0"


class RunContext:
    """Holds all paths and metadata for a single pipeline run.

    Directory layout::

        runs/{run_id}/
        ├── input/
        │   └── report.pdf
        ├── pages/            # PDF-rendered JPG pages
        ├── output/
        │   ├── step1_ocr/
        │   ├── step2_layoutlmv3/
        │   ├── step3_table_understanding/
        │   ├── step4_semantic_mapping/
        │   └── step5_visualization/
        │       └── overlay/
        ├── logs/
        ├── state.json
        └── events.jsonl
    """

    def __init__(self, run_id: str, project_root: str, run_root: str):
        self.run_id = run_id
        self.project_root = Path(project_root)
        self.run_root = Path(run_root)

        # Input
        self.input_dir = self.run_root / "input"
        self.input_pdf = self.input_dir / "report.pdf"

        # Pages (rendered images)
        self.pages_dir = self.run_root / "pages"

        # Output per step
        self.output_root = self.run_root / "output"
        self.step1_ocr_dir = self.output_root / "step1_ocr"
        self.step2_layout_dir = self.output_root / "step2_layoutlmv3"
        self.step3_table_dir = self.output_root / "step3_table_understanding"
        self.step4_semantic_dir = self.output_root / "step4_semantic_mapping"
        self.step5_viz_dir = self.output_root / "step5_visualization"
        self.step5_overlay_dir = self.step5_viz_dir / "overlay"

        # Logs
        self.logs_dir = self.run_root / "logs"

        # State & events
        self.state_path = self.run_root / "state.json"
        self.events_path = self.run_root / "events.jsonl"

        # Computed output paths (matches old contract for backward compat)
        self.ocr_json = self.step1_ocr_dir / "ocr_words.json"
        self.layout_labels_json = self.step2_layout_dir / "0_layoutlmv3_labels.json"
        self.layout_words_json = self.step2_layout_dir / "layout_words.json"
        self.all_tables_json = self.step3_table_dir / "all_tables.json"
        self.pending_scope_json = self.step3_table_dir / "pending_scope_classification.json"
        self.layout_blocks_json = self.step4_semantic_dir / "0_layoutlmv3_layout.json"
        self.scope_predictions_json = self.step4_semantic_dir / "scope_predictions.json"
        self.table_scope_json = self.step4_semantic_dir / "table_scope_predictions.json"
        self.unified_json = self.step4_semantic_dir / "all_results_unified.json"
        self.viz_index_html = self.step5_viz_dir / "index.html"
        self.viz_csv = self.step5_viz_dir / "scope_predictions_all.csv"

        # Model checkpoints (shared, not per-run)
        self.layoutlmv3_ckpt = str(self.project_root / "Output" / "2_Model_Layoutlmv3_Finetune" / "checkpoint-1000")
        self.climatebert_ckpt = str(self.project_root / "Output" / "4_Model_Classification_Scope" / "checkpoint-2178")
        self.tatr_model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
        self.layoutlmv3_model_id = "microsoft/layoutlmv3-base"

    # ─── Directory helpers ──────────────────────────────────────

    def ensure_dirs(self):
        """Create all run directories."""
        for d in [self.input_dir, self.pages_dir, self.output_root,
                  self.step1_ocr_dir, self.step2_layout_dir, self.step3_table_dir,
                  self.step4_semantic_dir, self.step5_viz_dir, self.step5_overlay_dir,
                  self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    def clean_output(self, step: str = None):
        """Remove output artifacts. If step is None, clean all outputs."""
        targets = {
            "convert": None,
            "ocr": self.step1_ocr_dir,
            "layout": self.step2_layout_dir,
            "tables": self.step3_table_dir,
            "scope": self.step4_semantic_dir,
            "visualize": self.step5_viz_dir,
        }
        if step:
            d = targets.get(step)
            if d and d.exists():
                shutil.rmtree(str(d))
                d.mkdir(parents=True)
        else:
            for d in targets.values():
                if d and d.exists():
                    shutil.rmtree(str(d))
            self.ensure_dirs()

    def clean_pages(self):
        """Remove all rendered page images."""
        for f in glob.glob(str(self.pages_dir / "*")):
            try:
                if os.path.isfile(f): os.remove(f)
                elif os.path.isdir(f): shutil.rmtree(f)
            except Exception:
                pass

    def save_state(self, data: dict):
        """Write state atomically."""
        tmp = str(self.state_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, str(self.state_path))

    def load_state(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def step_output_dir(self, step_index: int) -> Path:
        dirs = [self.step1_ocr_dir, self.step2_layout_dir, self.step3_table_dir,
                self.step4_semantic_dir, self.step4_semantic_dir, self.step5_viz_dir]
        return dirs[step_index] if 0 <= step_index < len(dirs) else self.output_root

    # ─── Input PDF helpers ──────────────────────────────────────

    def save_pdf(self, content: bytes, filename: str = None):
        """Save uploaded PDF under the stable path expected by pipeline steps."""
        safe = "".join(c for c in (filename or "report.pdf") if c.isalnum() or c in "._- ")[:100]
        if not safe.lower().endswith(".pdf"):
            safe += ".pdf"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        dest = self.input_pdf
        dest.write_bytes(content)
        if safe != self.input_pdf.name:
            (self.input_dir / safe).write_bytes(content)
        return dest

    def resolve_input_pdf(self) -> Path:
        """Return the uploaded PDF path, tolerating older runs without report.pdf."""
        if self.input_pdf.exists():
            return self.input_pdf
        candidates = sorted(self.input_dir.glob("*.pdf"))
        if candidates:
            self.input_pdf = candidates[0]
            return self.input_pdf
        return self.input_pdf


def run_context_from_args(args, project_root: str = None) -> RunContext:
    """Create RunContext from argparse namespace or simple dict.

    Expects ``args.run_id`` and optionally ``args.run_root``.
    If ``run_root`` is omitted, uses ``runs/{run_id}`` under project_root.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_id = getattr(args, "run_id", None) or f"run_{int(time.time())}"
    run_root = getattr(args, "run_root", None) or os.path.join(project_root, "runs", run_id)
    return RunContext(run_id, project_root, run_root)
