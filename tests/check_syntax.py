"""Quick syntax check on all modified files."""
import sys, os, ast

files = [
    'run_all.py',
    'ui/server.py',
    'ui/static/index.html',
    'pipeline_core/context.py',
    'pipeline_core/config.py',
    'pipeline_core/events.py',
    'pipeline_core/utils.py',
    '1.OCR_step/run_ocr_simple.py',
    '2.LayoutLMV3_step/inference_layoutlmv3.py',
    '3.Table_Understanding/run.py',
    '3.Table_Understanding/pipeline.py',
    '4.SemanticMapping/build_layout_blocks.py',
    '4.SemanticMapping/run_scope_inference.py',
    '4.SemanticMapping/visualize_results.py',
]

errors = []
for f in files:
    if f.endswith('.html'):
        # Basic HTML check
        content = open(f, encoding='utf-8').read()
        if '<html' not in content:
            errors.append(f"{f}: missing <html> tag")
        if '<script>' not in content:
            errors.append(f"{f}: missing <script> tag")
    else:
        try:
            with open(f, encoding='utf-8') as fh:
                ast.parse(fh.read())
        except SyntaxError as e:
            errors.append(f"{f}: {e}")

print(f"Checked {len(files)} files")
if errors:
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All files OK")
