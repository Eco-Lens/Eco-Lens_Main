# Semantic Mapping Pipeline

This folder contains a pipeline for ESG knowledge extraction, semantic mapping, dataset creation, and scope classification using LayoutLMv3 and ClimateBERT.

## Pipeline order

1. `ESG_Knowledge_Base.ipynb`
2. `layoutlmv3_inference.py`
3. `SemanticMapping_FAISS.ipynb`
4. `Create_Scope_Dataset.ipynb`
5. `balance_scope_dataset.py`
6. `Train_ClimateBERT_Scope.ipynb`
7. `Inference_ClimateBERT_Scope.ipynb`

## File summaries and data flow

### 1. `ESG_Knowledge_Base.ipynb`
- Purpose: build an ESG knowledge base from raw PDF reports and allow semantic search.
- Main steps:
  - unzip `ESG_KB.zip` and read PDF files with `pymupdf`/`fitz`
  - extract page text into a document/page JSON structure
  - clean text and chunk each page into ESG text chunks
  - save chunks to `output/esg_chunks.json`
  - embed chunks with `climatebert/distilroberta-base-climate-f`
  - save embeddings to `output/embeddings.npy`
  - normalize embeddings and build a FAISS index
  - save `faiss/esg.index` and metadata to `faiss/metadata.json`
  - run example semantic search queries against the FAISS knowledge base
- Input:
  - `ESG_KB.zip` containing ESG report PDF files
- Outputs:
  - `output/raw_documents.json` (document/page text structure)
  - `output/esg_chunks.json` (chunked ESG text entries)
  - `output/embeddings.npy` (embedding matrix, shape = [num_chunks, dim])
  - `faiss/esg.index` (FAISS similarity index)
  - `faiss/metadata.json` (chunk metadata matching index positions)
- Data structure:
  - raw document: list of documents, each with `document`, `file`, `num_pages`, `pages`
  - chunk: `{chunk_id, document, standard, page, local_chunk, num_words, num_chars, text}`
  - FAISS metadata: list of chunk records with same fields plus text

### 2. `layoutlmv3_inference.py`
- Purpose: infer LayoutLMv3 token labels for OCR text and structure text blocks.
- Main steps:
  - load fine-tuned LayoutLMv3 token classification checkpoint
  - read OCR JSON input from `Output/1_OCR/0_ocr_words.json`
  - for each image in `valid/`, normalize bounding boxes and chunk words into groups of 60
  - run LayoutLMv3 inference and map token predictions back to words via majority vote
  - sort words and group them into text/paragraph blocks
  - save structured page/block output to `test/0_layoutlmv3_layout.json`
- Input:
  - `Output/1_OCR/0_ocr_words.json` (OCR words, bboxes, text, confidence)
  - image files under `valid/`
  - LayoutLMv3 checkpoint: `Output/2_Model_Layoutlmv3_Finetune/checkpoint-1000`
- Output:
  - `test/0_layoutlmv3_layout.json`
- Data structure:
  - top-level mapping image name → page record
  - page record: `{page, num_words, num_blocks, block_summary, blocks}`
  - block: `{type, labels, text, bbox, confidence, num_lines, num_words, words}`
  - word: `{text, bbox, label, confidence}`

### 3. `SemanticMapping_FAISS.ipynb`
- Purpose: semantically map layout text blocks to ESG knowledge base chunks.
- Main steps:
  - load layout inference JSON `0_layoutlmv3_layout.json`
  - extract text/figure blocks with metadata from layout pages
  - encode each block text using ClimateBERT embeddings
  - search the prebuilt FAISS knowledge base (`esg.index`) for top matches
  - attach top match list to each block and select the best semantic match
  - save `semantic_blocks.json`
- Input:
  - `0_layoutlmv3_layout.json`
  - `faiss/esg.index`
  - `faiss/metadata.json`
- Output:
  - `semantic_output/semantic_blocks.json`
- Data structure:
  - semantic block: `{block_id, page, bbox, type, confidence, text, top_matches, semantic_standard, semantic_document, semantic_score}`
  - each `top_matches` item includes metadata from the FAISS KB plus `score`

### 4. `Create_Scope_Dataset.ipynb`
- Purpose: create a labeling-ready dataset from semantic mapping blocks.
- Main steps:
  - load `semantic_output/semantic_blocks.json`
  - inspect one sample block structure
  - build a DataFrame with block metadata and semantic match fields
  - add an empty `scope` column for manual annotation
  - produce a label-ready table for download/editing
- Input:
  - `semantic_output/semantic_blocks.json`
- Output:
  - interactive dataset in notebook, later saved/downloaded as `label_scope_dataset.xlsx`
- Data structure:
  - `block_id`, `page`, `text`, `type`, `confidence`, `semantic_standard`, `semantic_document`, `semantic_score`, `matched_chunk`, `matched_page`, `matched_document`, `scope`

### 5. `balance_scope_dataset.py`
- Purpose: clean, score, and rebalance the manually labeled scope dataset.
- Main steps:
  - load `label_scope_dataset.xlsx`
  - ensure required columns exist and fill missing scope values with `Other`
  - remove rows with low-quality or meaningless text
  - deduplicate text and block IDs while keeping higher-quality rows
  - compute a custom quality score based on semantic score, ESG keywords, text length, and label type
  - downsample/oversample classes to target sizes
  - save balanced dataset files
- Input:
  - `label_scope_dataset.xlsx`
- Outputs:
  - `scope_dataset_balanced.xlsx`
  - `scope_dataset_balanced.csv`
- Data structure:
  - same columns as input plus a computed `quality_score` during processing
  - target class counts: `Scope 1=1000`, `Scope 2=1000`, `Scope 3=1700`, `Other=1700`
  - for real: `Scope 1=1000`, `Scope 2=1000`, `Scope 3=1138`, `Other=1700`

### 6. `Train_ClimateBERT_Scope.ipynb`
- Purpose: fine-tune a ClimateBERT classifier on the balanced scope dataset.
- Main steps:
  - load `scope_dataset_balanced.csv`
  - keep only `text` and `scope` columns and clean empty text
  - encode labels with `label2id = {Other:0, Scope 1:1, Scope 2:2, Scope 3:3}`
  - split into train/validation/test sets stratified by label
  - convert Pandas DataFrames to HuggingFace `Dataset`
  - load `climatebert/distilroberta-base-climate-f` tokenizer and model
  - tokenize text with `max_length=256`
  - define metrics: accuracy, precision, recall, macro/weighted F1
  - train across multiple random seeds, saving best model by validation macro F1
  - save best model to `ClimateBERT_Scope/best_scope_classifier`
  - save `label_mapping.json`, `training_config.json`, and training summary
- Input:
  - `scope_dataset_balanced.csv`
- Outputs:
  - `ClimateBERT_Scope/best_scope_classifier/` (model + tokenizer)
  - `ClimateBERT_Scope/best_scope_classifier/label_mapping.json`
  - training summary/output saved under model directory
- Data structure:
  - HF dataset with tokenized fields `input_ids`, `attention_mask`, and labels
  - metrics dictionaries for train/eval/test

### 7. `Inference_ClimateBERT_Scope.ipynb`
- Purpose: apply the trained scope classifier to layout blocks.
- Main steps:
  - load best model from `ClimateBERT_Scope/best_scope_classifier`
  - load label mapping from `label_mapping.json`
  - read layout blocks from `0_layoutlmv3_layout.json`
  - predict `scope` label, confidence, and probabilities for each block text
  - save predictions to `SemanticMapping_Inference/scope_predictions.csv` and `scope_predictions.json`
- Input:
  - `ClimateBERT_Scope/best_scope_classifier`
  - `0_layoutlmv3_layout.json`
- Output:
  - `SemanticMapping_Inference/scope_predictions.csv`
  - `SemanticMapping_Inference/scope_predictions.json`
- Data structure:
  - each prediction record includes original block fields plus `scope`, `scope_id`, `confidence`, and `probabilities`

## Notes

- `label_scope_dataset.xlsx` is created by manual labeling in `Create_Scope_Dataset.ipynb` and then used by `balance_scope_dataset.py`.
- `reason.md` documents the balancing rationale used in `balance_scope_dataset.py`.
- `0_layoutlmv3_layout.json` is the output of `layoutlmv3_inference.py` and is reused by later semantic mapping and inference steps.
- `ClimateBERT_Scope/` contains the trained classifier and artifacts used for inference.
