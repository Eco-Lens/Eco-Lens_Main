"""
run.py — Table Understanding pipeline

Usage:
    python "3.Table_Understanding/run.py"

Prerequisites:
    python "1.OCR_step/run_ocr_simple.py" --root "test" --out_json "test/0_ocr_words.json"
    python "2.LayoutLMV3_step/inference_layoutlmv3.py"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; import torchvision

from pipeline import run

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

run(
    ocr_path=os.path.join(BASE, "test", "output", "step1_ocr", "0_ocr_words.json"),
    labels_path=os.path.join(BASE, "test", "output", "step2_layoutlmv3", "0_layoutlmv3_labels.json"),
    image_root=os.path.join(BASE, "test"),
    out_dir=os.path.join(BASE, "test", "output", "step3_table_understanding"),
)
