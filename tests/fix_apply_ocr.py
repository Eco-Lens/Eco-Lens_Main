# Fix inference_layoutlmv3.py - remove apply_ocr=False
lines = open('2.LayoutLMV3_step/inference_layoutlmv3.py', 'r').readlines()
for i, line in enumerate(lines):
    if 'apply_ocr=False' in line:
        lines[i] = line.replace(', apply_ocr=False', '')
        print(f'Fixed line {i+1}: {lines[i].strip()}')
open('2.LayoutLMV3_step/inference_layoutlmv3.py', 'w').writelines(lines)
print('Done')
