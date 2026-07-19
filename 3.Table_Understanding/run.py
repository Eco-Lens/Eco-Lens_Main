"""
run.py — Table Understanding entry point.

Usage:
    python "3.Table_Understanding/run.py" \\
        --ocr-json runs/{run_id}/output/step1_ocr/ocr_words.json \\
        --labels-json runs/{run_id}/output/step2_layoutlmv3/0_layoutlmv3_labels.json \\
        --image-dir runs/{run_id}/pages \\
        --out-dir runs/{run_id}/output/step3_table_understanding
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_core.config import SCHEMA_VERSION
from pipeline_core.utils import atomic_write_json

from pipeline import run as pipeline_run


def main():
    ap = argparse.ArgumentParser(description="Table Understanding Pipeline")
    ap.add_argument("--ocr-json", required=True, help="OCR words JSON")
    ap.add_argument("--labels-json", required=True, help="LayoutLMv3 labels JSON")
    ap.add_argument("--image-dir", required=True, help="Page images directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--run-id", default=None, help="Run ID (for metadata)")
    args = ap.parse_args()

    run_id = args.run_id or os.environ.get("RUN_ID", "unknown")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"OCR JSON:    {args.ocr_json}")
    print(f"Labels JSON: {args.labels_json}")
    print(f"Image dir:   {args.image_dir}")
    print(f"Out dir:     {args.out_dir}")

    # Call pipeline.run() — it already accepts function parameters
    pipeline_run(
        ocr_path=args.ocr_json,
        labels_path=args.labels_json,
        image_root=args.image_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
