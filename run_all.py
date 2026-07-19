#!/usr/bin/env python3
"""run_all.py — Eco-Lens pipeline orchestrator (run-isolated, CLI-driven).

Usage:
    python run_all.py --run-id my_run
    python run_all.py --fresh
    python run_all.py --resume
    python run_all.py --run-id my_run --step 2  # resume from step 2
"""

import sys, os, time, subprocess, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core.context import RunContext
from pipeline_core.config import STEPS


def run_step(ctx: RunContext, step_index: int, step_def: dict, scripts: list):
    """Execute one pipeline step via subprocess."""
    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["RUN_ID"] = ctx.run_id
    env["RUN_ROOT"] = str(ctx.run_root)

    # Build command: script path + args
    cmd = [sys.executable, scripts[step_index]]
    if step_index == 0:  # OCR
        cmd += ["--image-dir", str(ctx.pages_dir), "--out-json", str(ctx.ocr_json)]
    elif step_index == 1:  # LayoutLMv3 labels
        cmd += ["--ocr-json", str(ctx.ocr_json), "--image-dir", str(ctx.pages_dir),
                "--out-labels", str(ctx.layout_labels_json),
                "--out-layout", str(ctx.layout_words_json)]
    elif step_index == 2:  # Table Understanding
        cmd += ["--ocr-json", str(ctx.ocr_json), "--labels-json", str(ctx.layout_labels_json),
                "--image-dir", str(ctx.pages_dir), "--out-dir", str(ctx.step3_table_dir)]
    elif step_index == 3:  # Block grouping (reads layout_words.json, no model load)
        cmd += ["--layout-json", str(ctx.layout_words_json), "--image-dir", str(ctx.pages_dir),
                "--out-json", str(ctx.layout_blocks_json)]
    elif step_index == 4:  # Scope classification
        cmd += ["--layout-json", str(ctx.layout_blocks_json), "--tables-json", str(ctx.all_tables_json),
                "--image-dir", str(ctx.pages_dir), "--out-dir", str(ctx.step4_semantic_dir)]
    elif step_index == 5:  # Visualization
        cmd += ["--unified-json", str(ctx.unified_json), "--layout-json", str(ctx.layout_blocks_json),
                "--tables-json", str(ctx.all_tables_json),
                "--image-dir", str(ctx.pages_dir), "--out-dir", str(ctx.step5_viz_dir)]

    step_name = step_def["name"]
    t0 = time.time()
    print(f"\n{'='*60}\n[{step_index}] {step_name}\n{'='*60}")

    result = subprocess.run(cmd, cwd=str(ctx.project_root), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=3600)
    elapsed = time.time() - t0

    # Print output
    for line in result.stdout.decode("utf-8", errors="replace").split("\n"):
        if line.strip():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  [FAIL] {step_name} failed (rc={result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    print(f"  [OK] {step_name} completed in {elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Eco-Lens Pipeline Orchestrator")
    ap.add_argument("--run-id", default=None, help="Run identifier (default: auto)")
    ap.add_argument("--run-root", default=None, help="Run directory (default: runs/{run_id})")
    ap.add_argument("--project-root", default=None, help="Project root (default: auto-detect)")
    ap.add_argument("--fresh", action="store_true", help="Clean outputs before starting")
    ap.add_argument("--resume", action="store_true", help="Resume pipeline from last incomplete step")
    ap.add_argument("--step", type=int, default=0, help="Start from step index (0-based)")
    ap.add_argument("--pdf", default=None, help="Path to PDF file (required for step 0)")
    args = ap.parse_args()

    project_root = args.project_root or os.path.dirname(os.path.abspath(__file__))
    run_id = args.run_id or f"run_{int(time.time())}"
    run_root = args.run_root or os.path.join(project_root, "runs", run_id)

    ctx = RunContext(run_id, project_root, run_root)
    ctx.ensure_dirs()

    print(f"Run ID:    {ctx.run_id}")
    print(f"Run root:  {ctx.run_root}")
    print(f"Project:   {ctx.project_root}")

    # Step 0: PDF rendering (only if PDF provided)
    scripts = [
        os.path.join(project_root, "1.OCR_step", "run_ocr_simple.py"),
        os.path.join(project_root, "2.LayoutLMV3_step", "inference_layoutlmv3.py"),
        os.path.join(project_root, "3.Table_Understanding", "run.py"),
        os.path.join(project_root, "4.SemanticMapping", "build_layout_blocks.py"),
        os.path.join(project_root, "4.SemanticMapping", "run_scope_inference.py"),
        os.path.join(project_root, "4.SemanticMapping", "visualize_results.py"),
    ]

    start_step = 0
    resume_info = "fresh"

    if args.resume:
        state = ctx.load_state()
        completed = state.get("completed_steps", [])
        # If step 0 (PDF render) is not completed, we need the PDF
        if 0 not in completed and not args.pdf:
            print("--resume: step 0 not completed. Provide --pdf to render.")
            sys.exit(1)
        start_step = max(completed) + 1 if completed else 0
        if start_step >= len(STEPS):
            print("All steps already completed. Use --fresh to re-run.")
            return
        resume_info = f"resume from step {start_step}"
        print(f"Resume: completed {completed}, resuming at step {start_step}")

    if args.fresh or (not args.resume and start_step == 0):
        ctx.clean_output()
        ctx.clean_pages()
        resume_info = "fresh"

    print(f"Mode: {resume_info}")

    # Step 0: PDF → Images (if PDF provided and step 0 not done)
    if start_step == 0:
        if not args.pdf:
            print("No --pdf provided. Skipping PDF rendering (use --pdf for full pipeline).")
            # If pages dir already has images, proceed
            pages = list(ctx.pages_dir.glob("*.jpg"))
            if not pages:
                print("No pages found. Provide --pdf to render pages.")
                sys.exit(1)
            print(f"Found {len(pages)} existing page images, skipping PDF rendering.")
        else:
            if not os.path.exists(args.pdf):
                print(f"PDF not found: {args.pdf}")
                sys.exit(1)
            pdf_data = open(args.pdf, "rb").read()
            ctx.save_pdf(pdf_data, os.path.basename(args.pdf))
            print(f"PDF saved: {ctx.input_pdf} ({len(pdf_data)} bytes)")

            import fitz
            doc = fitz.open(str(ctx.input_pdf))
            num_pages = len(doc)
            print(f"Rendering {num_pages} pages...")
            for i in range(num_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=200)
                fname = f"page_{i+1:03d}.jpg"
                pix.save(str(ctx.pages_dir / fname))
                if (i+1) % 5 == 0 or i == num_pages - 1:
                    print(f"  Page {i+1}/{num_pages}")
            doc.close()
            print(f"Rendered {num_pages} pages to {ctx.pages_dir}")

        ctx.ensure_dirs()
        state = ctx.load_state()
        completed = state.get("completed_steps", [])
        if 0 not in completed:
            completed.append(0)
            ctx.save_state({"completed_steps": completed, "run_id": ctx.run_id})
        start_step = 1

    # Steps 1-6 (si = STEPS index starting from 1, script_idx = 0-based scripts index)
    for si in range(start_step, len(STEPS)):
        script_idx = si - 1
        # Don't clean scope dir (shared with blocks step)
        if STEPS[si]["id"] not in ("scope",):
            ctx.clean_output(STEPS[si]["id"])
        run_step(ctx, script_idx, STEPS[si], scripts)
        state = ctx.load_state()
        completed = state.get("completed_steps", [])
        if si not in completed:
            completed.append(si)
        ctx.save_state({"completed_steps": completed, "run_id": ctx.run_id})

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Run root: {ctx.run_root}")
    print(f"HTML report: {ctx.viz_index_html}")
    print(f"CSV export:  {ctx.viz_csv}")


if __name__ == "__main__":
    main()
