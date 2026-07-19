"""
layoutlmv3_inference.py (LEGACY WRAPPER)

This file previously loaded LayoutLMv3 and re-ran inference.
It now delegates to ``build_layout_blocks.py`` which reads the
rich layout artifact from Step 2 (layout_words.json).

Kept for backward compatibility with ``run_all.py`` and older scripts.
Usage is unchanged:
    python "4.SemanticMapping/layoutlmv3_inference.py"
"""
import sys, os

# Delegate to build_layout_blocks with paths derived from legacy defaults
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline_core.context import RunContext
from pipeline_core.config import STEPS

if __name__ == "__main__":
    run_id = os.environ.get("RUN_ID", f"run_{int(__import__('time').time())}")
    run_root = os.environ.get("RUN_ROOT", os.path.join(PROJECT_ROOT, "runs", run_id))
    ctx = RunContext(run_id, PROJECT_ROOT, run_root)

    # If the new layout artifact doesn't exist, fall back to legacy paths
    layout_json = str(ctx.layout_words_json)
    if not os.path.exists(layout_json):
        print("Layout artifact not found, falling back to legacy paths...")
        layout_json = str(ctx.project_root / "test" / "output" / "step2_layoutlmv3" / "layout_words.json")
        if not os.path.exists(layout_json):
            layout_json = str(ctx.project_root / "test" / "output" / "step2_layoutlmv3" / "0_layoutlmv3_labels.json")

    out_json = str(ctx.layout_blocks_json)
    if not os.path.exists(os.path.dirname(out_json)):
        out_json = str(ctx.project_root / "test" / "output" / "step4_semantic_mapping" / "0_layoutlmv3_layout.json")

    image_dir = str(ctx.pages_dir)
    if not os.path.exists(image_dir):
        image_dir = str(ctx.project_root / "test")

    print(f"Delegating to build_layout_blocks.py...")
    print(f"  Layout JSON: {layout_json}")
    print(f"  Output:      {out_json}")

    # Run build_layout_blocks as subprocess with same Python interpreter
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "4.SemanticMapping", "build_layout_blocks.py"),
        "--layout-json", layout_json,
        "--image-dir", image_dir,
        "--out-json", out_json,
        "--run-id", run_id,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)
