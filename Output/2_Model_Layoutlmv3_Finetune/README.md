# LayoutLMv3 Fine-tuned Model

This folder stores the download instruction for the fine-tuned LayoutLMv3 model checkpoint.

Because the model checkpoint is large, it is not stored directly in this Git repository.  
The checkpoint is hosted on Hugging Face.

## Model checkpoint

File name:

```text
checkpoint-1000-20260708T070121Z-3-001.zip
```

File size: approximately 1.23 GB.

Hugging Face repository:

```text
https://huggingface.co/SRegit/Layoutlmv3_Finetune_Output
```

---

## Option 1: Download directly from browser

You can download the model checkpoint using this direct link:

```text
https://huggingface.co/SRegit/Layoutlmv3_Finetune_Output/resolve/main/checkpoint-1000-20260708T070121Z-3-001.zip
```

After downloading, place the ZIP file into this folder:

```text
Eco-Lens_Main\Output\2_Model_Layoutlmv3_Finetune
```

Expected structure:

```text
Eco-Lens_Main/
└── Output/
    └── 2_Model_Layoutlmv3_Finetune/
        ├── README.md
        └── checkpoint-1000-20260708T070121Z-3-001.zip
```

---

## Option 2: Download with Hugging Face CLI

Install Hugging Face Hub CLI:

```bash
pip install -U huggingface_hub
```

Then download the checkpoint directly into the model folder:

```bash
hf download SRegit/Layoutlmv3_Finetune_Output checkpoint-1000-20260708T070121Z-3-001.zip --local-dir Eco-Lens_Main/Output/2_Model_Layoutlmv3_Finetune
```

Alternative command from Hugging Face:

```bash
hf download hf://SRegit/Layoutlmv3_Finetune_Output/checkpoint-1000-20260708T070121Z-3-001.zip
```

If the repository is private, login first:

```bash
hf auth login
```

---

## Extract the checkpoint

After downloading, extract the ZIP file.

On Windows PowerShell:

```powershell
Expand-Archive -Path "Eco-Lens_Main\Output\2_Model_Layoutlmv3_Finetune\checkpoint-1000-20260708T070121Z-3-001.zip" -DestinationPath "Eco-Lens_Main\Output\2_Model_Layoutlmv3_Finetune\checkpoint-1000"
```

On Linux/macOS:

```bash
unzip Eco-Lens_Main/Output/2_Model_Layoutlmv3_Finetune/checkpoint-1000-20260708T070121Z-3-001.zip -d Eco-Lens_Main/Output/2_Model_Layoutlmv3_Finetune/checkpoint-1000
```

Expected structure after extraction:

```text
Eco-Lens_Main/
└── Output/
    └── 2_Model_Layoutlmv3_Finetune/
        ├── README.md
        ├── checkpoint-1000-20260708T070121Z-3-001.zip
        └── checkpoint-1000/
            └── ...
```

---

## Load the model

After extracting the checkpoint, load it with Hugging Face Transformers:

```python
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

model_dir = "Eco-Lens_Main/Output/2_Model_Layoutlmv3_Finetune/checkpoint-1000"

processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
```

---

## Notes

This checkpoint is the fine-tuned LayoutLMv3 model output.

To reproduce correct inference results, make sure to use the same preprocessing pipeline as training, including:

- OCR text extraction
- Bounding box format
- Bounding box normalization
- Label mapping
- Image size handling
- Token alignment logic