"""
run_all.py — Run the complete Eco-Lens pipeline (5 steps).

Usage:
    python run_all.py

Prerequisites:
    pip install -r requirements.txt  (if applicable)
    test/*.jpg images exist
"""
import sys, os, time, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIRS = [
    "test/output/step1_ocr",
    "test/output/step2_layoutlmv3",
    "test/output/step3_table_understanding",
    "test/output/step4_semantic_mapping",
    "test/output/step5_visualization",
]

STEPS = [
    {
        "name": "Step 1 — OCR (PaddleOCR)",
        "cmd": [sys.executable, "1.OCR_step/run_ocr_simple.py",
                "--root", "test",
                "--out_json", "test/output/step1_ocr/0_ocr_words.json"],
    },
    {
        "name": "Step 2 — LayoutLMv3 (word-level labels)",
        "cmd": [sys.executable, "2.LayoutLMV3_step/inference_layoutlmv3.py"],
        "env": {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
    },
    {
        "name": "Step 3 — Table Understanding (TATR)",
        "cmd": [sys.executable, "3.Table_Understanding/run.py"],
        "env": {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
    },
    {
        "name": "Step 4a — Block grouping",
        "cmd": [sys.executable, "4.SemanticMapping/layoutlmv3_inference.py"],
        "env": {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
    },
    {
        "name": "Step 4b — Scope classification (ClimateBERT)",
        "cmd": [sys.executable, "4.SemanticMapping/run_scope_inference.py"],
        "env": {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
    },
    {
        "name": "Step 5 — Visualization",
        "cmd": [sys.executable, "4.SemanticMapping/visualize_results.py"],
    },
]


def run():
    for d in OUTPUT_DIRS:
        os.makedirs(os.path.join(BASE, d), exist_ok=True)

    t_start = time.time()
    for step in STEPS:
        print(f"\n{'='*60}")
        print(f"{step['name']}")
        print(f"{'='*60}")
        env = os.environ.copy()
        env.update(step.get("env", {}))
        t0 = time.time()
        result = subprocess.run(step["cmd"], cwd=BASE, env=env)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\nFAILED after {elapsed:.0f}s — exit code {result.returncode}")
            sys.exit(1)
        print(f"Completed in {elapsed:.0f}s")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline finished in {total:.0f}s")
    print(f"Output: {os.path.join(BASE, 'test', 'output', 'step5_visualization', 'index.html')}")


if __name__ == "__main__":
    run()
