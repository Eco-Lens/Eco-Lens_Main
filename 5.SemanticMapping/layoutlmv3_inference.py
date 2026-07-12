"""
inference_layoutlmv3.py
------------------------
Load LayoutLMv3 model từ checkpoint fine-tuned.
Inference bằng LayoutLMv3Processor (image + words + boxes).
Chunk words thành từng nhóm 60 words (no overlap) để tránh max_length=512.

Cách dùng:
    python "5.SemanticMapping/layoutlmv3_inference.py"
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import Counter
from PIL import Image

import transformers.utils.generic as generic
generic.is_tf_available = lambda: False

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

CKPT = os.path.join("Output", "2_Model_Layoutlmv3_Finetune", "checkpoint-1000")
OCR_JSON = os.path.join("Output", "1_OCR", "0_ocr_words.json")
LAYOUT_JSON_OUT = os.path.join("test", "0_layoutlmv3_layout.json")
IMAGE_DIR = "valid"

CHUNK_SIZE = 60
MAX_LEN = 512

ID2LABEL = {
    0: "O", 1: "chart", 2: "figure", 3: "footer", 4: "header",
    5: "ignore", 6: "table", 7: "table_text", 8: "text", 9: "toc",
}


def clean_bbox(bbox, w, h):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    # Normalize to 0-1000
    return [round(1000 * x0 / w), round(1000 * y0 / h),
            round(1000 * x1 / w), round(1000 * y1 / h)]

def sort_words_reading_order(words):
    """
    Reading order

    top -> bottom

    left -> right

    dùng center_y giúp OCR lệch ít
    """

    for w in words:

        x0, y0, x1, y1 = w["bbox"]

        w["cx"] = (x0 + x1) / 2

        w["cy"] = (y0 + y1) / 2

    return sorted(

        words,

        key=lambda w:(
            round(w["cy"]/8),
            w["bbox"][0]
        )
    )

def reconstruct_lines(words,
                      y_threshold=10):

    if len(words)==0:
        return []

    words = sort_words_reading_order(words)

    lines=[]

    current=[words[0]]

    current_y=(words[0]["bbox"][1] + words[0]["bbox"][3])/2

    for w in words[1:]:
        same_line = abs(w["cy"] - current_y) <= y_threshold

        if same_line:
            current.append(w)
            current_y = np.mean([x["cy"] for x in current])
        else:

            current.sort(key=lambda x: x["bbox"][0])
            lines.append(current)

            current = [w]
            current_y = w["cy"]

    current.sort(

        key=lambda x:x["bbox"][0]

    )

    lines.append(current)

    return lines

def merge_lines_to_paragraphs(
    lines,
    line_gap_ratio=1.5,
    indent_ratio=0.05,
):

    if len(lines) == 0:
        return []

    paragraphs = []

    current = [lines[0]]

    for line in lines[1:]:

        prev = current[-1]

        line_height = np.mean([
            w["bbox"][3] - w["bbox"][1]
            for w in prev
        ])

        line_gap = max(12, line_height * line_gap_ratio)

        page_width = 1000
        indent_threshold = page_width * indent_ratio

    for line in lines[1:]:

        prev=current[-1]

        prev_bottom=max(

            w["bbox"][3]

            for w in prev

        )

        line_top=min(

            w["bbox"][1]

            for w in line

        )

        gap=line_top-prev_bottom

        prev_left=min(

            w["bbox"][0]

            for w in prev

        )

        line_left=min(

            w["bbox"][0]

            for w in line

        )

        if (

            gap<=line_gap

            and

            abs(prev_left-line_left)

            <=indent_threshold

        ):

            current.append(line)

        else:

            paragraphs.append(current)

            current=[line]

    paragraphs.append(current)

    return paragraphs

def group_text_blocks(words):

    valid_labels={

        "text",

        "figure"

    }

    words=[

        w

        for w in words

        if w["label"] in valid_labels

    ]

    if len(words)==0:

        return []

    lines=reconstruct_lines(words)

    paragraphs=merge_lines_to_paragraphs(lines)

    output=[]

    for para in paragraphs:

        para_words=[]
        para.sort(key=lambda line:min(w["bbox"][1] for w in line))
        for line in para:

            para_words.extend(line)

        text="\n".join(

            " ".join(

                w["text"]

                for w in sorted(

                    line,

                    key=lambda x:x["bbox"][0]

                )

            )

            for line in para

        )

        bbox=[

            min(

                w["bbox"][0]

                for w in para_words

            ),

            min(

                w["bbox"][1]

                for w in para_words

            ),

            max(

                w["bbox"][2]

                for w in para_words

            ),

            max(

                w["bbox"][3]

                for w in para_words

            )

        ]

        confidence=float(

            np.mean(

                [

                    w["confidence"]

                    for w in para_words

                ]

            )

        )

        labels = sorted(set(w["label"] for w in para_words))

        output.append({

            "type": "mixed" if len(labels) > 1 else labels[0],

            "labels": labels,

            "text":text,

            "bbox":bbox,

            "confidence":confidence,

            "num_lines":len(para),

            "num_words":len(para_words),

            "words":para_words

        })

    return output

def main():
    t0 = time.time()
    print("Loading processor + model...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(CKPT)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Loaded on {device} in {time.time()-t0:.0f}s")

    with open(OCR_JSON, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    results = {}

    for img_name, words_list in sorted(ocr_data.items()):
        if not words_list:
            results[img_name] = {
                "page": img_name,
                "num_words": 0,
                "num_blocks": 0,
                "block_summary": {},
                "blocks": []
            }
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            results[img_name] = {
                "page": img_name,
                "num_words": 0,
                "num_blocks": 0,
                "block_summary": {},
                "blocks": []
            }
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        n_words = len(words_list)

        texts = [x["text"] for x in words_list]
        bboxes_norm = [clean_bbox(x["bbox"], w, h) for x in words_list]

        # Chunk: CHUNK_SIZE words per chunk, no overlap
        final_labels = [None] * n_words
        final_scores = [0.0] * n_words

        chunks_info = []
        for start in range(0, n_words, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_words)
            chunk_words = texts[start:end]
            chunk_boxes = bboxes_norm[start:end]

            encoding = processor(
                images=img,
                text=chunk_words,
                boxes=chunk_boxes,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )

            word_ids = encoding.word_ids(batch_index=0)
            model_in = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**model_in)

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            pred_scores_token = torch.max(probs, dim=-1).values.detach().cpu().numpy()

            # Map token predictions → word (majority vote)
            word_pred_ids = {}
            word_pred_scores = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                word_pred_ids.setdefault(word_id, []).append(int(pred_ids[token_idx]))
                word_pred_scores.setdefault(word_id, []).append(float(pred_scores_token[token_idx]))

            for local_idx in range(len(chunk_words)):
                global_idx = start + local_idx
                if local_idx not in word_pred_ids:
                    final_labels[global_idx] = "O"
                    final_scores[global_idx] = 0.0
                else:
                    majority_id = Counter(word_pred_ids[local_idx]).most_common(1)[0][0]
                    final_labels[global_idx] = ID2LABEL.get(majority_id, "O")
                    final_scores[global_idx] = float(np.mean(word_pred_scores[local_idx]))

            chunks_info.append({"start": start, "end": end, "num": end - start})

        final_labels = [l if l is not None else "O" for l in final_labels]

        word_objects=[]

        for i in range(n_words):

            word_objects.append({

                "text":texts[i],

                "bbox":bboxes_norm[i],

                "label":final_labels[i],

                "confidence":final_scores[i]

            })

        word_objects=sort_words_reading_order(

            word_objects

        )

        blocks=group_text_blocks(

            word_objects

        )

        block_counter = Counter(block["type"] for block in blocks)

        results[img_name]={

            "page":img_name,

            "num_words":len(word_objects),

            "num_blocks":len(blocks),

            "block_summary":dict(block_counter),

            "blocks":blocks

        }
        print(
            f"{img_name}: "
            f"{len(blocks)} blocks "
            f"{dict(block_counter)} "
            f"({len(chunks_info)} chunks)"
        )

    with open(LAYOUT_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved layout json to {LAYOUT_JSON_OUT}")


if __name__ == "__main__":
    main()
