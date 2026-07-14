import os
import re
import random
from collections import Counter
import pandas as pd
import numpy as np

random.seed(42)

INPUT_PATH = r"E:\Capstone\Eco-Lens_Main\5.SemanticMapping\label_scope_dataset.xlsx"
OUTPUT_XLSX = r"E:\Capstone\Eco-Lens_Main\5.SemanticMapping\scope_dataset_balanced.xlsx"
OUTPUT_CSV = r"E:\Capstone\Eco-Lens_Main\5.SemanticMapping\scope_dataset_balanced.csv"
REASON_PATH = r"E:\Capstone\Eco-Lens_Main\5.SemanticMapping\reason.md"

REQUIRED_COLUMNS = [
    "block_id",
    "page",
    "bbox",
    "type",
    "confidence",
    "semantic_standard",
    "semantic_document",
    "semantic_score",
    "matched_chunk",
    "matched_page",
    "matched_document",
    "text",
    "scope",
]

ESG_KEYWORDS = [
    "emission", "emissions", "ghg", "greenhouse", "carbon", "climate", "energy",
    "electricity", "diesel", "fuel", "travel", "waste", "water", "sustainability",
    "net zero", "renewable", "governance", "strategy", "policy", "environment",
    "scope", "supply chain", "procurement", "logistics", "commuting", "fleet",
    "steam", "heat", "cooling", "power"
]

OTHER_BAD_PATTERNS = [
    r"table of contents", r"copyright", r"all rights reserved", r"page number",
    r"header", r"footer", r"company address", r"contact us", r"www\.", r"http",
    r"logo", r"chapter", r"contents", r"index", r"title", r"slogan", r"message from",
    r"chairman", r"ceo", r"president", r"board", r"management", r"governance", r"strategy"
]


def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    return s


def is_meaningful_text(text):
    t = normalize_text(text)
    if not t:
        return False
    if len(t) < 3:
        return False
    if sum(c.isalpha() for c in t) < 2:
        return False
    if re.fullmatch(r"[^a-z0-9]+", t):
        return False
    if any(re.search(p, t) for p in OTHER_BAD_PATTERNS):
        return False
    # heavily broken OCR-like strings
    if re.search(r"([a-z]{1,2}\s){6,}", t):
        return False
    if t.count(".") > 8 and len(t.split()) < 5:
        return False
    return True


def quality_score(row):
    txt = normalize_text(row.get("text", ""))
    mc = normalize_text(row.get("matched_chunk", ""))
    comb = f"{txt} {mc}"
    score = 0.0

    if pd.notna(row.get("semantic_score")):
        try:
            score += float(row["semantic_score"]) * 2.0
        except Exception:
            pass

    if len(txt.split()) >= 6:
        score += 0.2
    elif len(txt.split()) >= 3:
        score += 0.1

    if any(k in comb for k in ESG_KEYWORDS):
        score += 0.35

    # stronger signal for carbon/emissions phrases
    if any(k in comb for k in ["emission", "emissions", "ghg", "carbon", "greenhouse", "climate", "energy", "electricity"]):
        score += 0.5

    # prefer text blocks over figure-only lines
    if str(row.get("type", "")).strip().lower() == "text":
        score += 0.15
    elif str(row.get("type", "")).strip().lower() == "figure":
        score -= 0.15

    # slightly reward matched chunk relevance
    if len(mc.split()) >= 3 and any(k in mc for k in ESG_KEYWORDS):
        score += 0.2

    # penalize very short or low information rows
    if len(txt) <= 8:
        score -= 0.3
    if len(txt) <= 20 and not any(k in txt for k in ["emission", "emissions", "scope", "energy", "electricity", "travel", "waste", "water", "carbon"]):
        score -= 0.2

    # priority for Other to keep ESG-like content and avoid page/header noise
    if row.get("scope") == "Other":
        if any(k in comb for k in ["climate", "energy", "emission", "sustainability", "renewable", "net zero", "strategy", "policy", "governance"]):
            score += 0.4
        if any(k in txt for k in ["waste", "water", "travel", "supply chain", "commuting", "procurement", "logistics"]):
            score += 0.3
        if not any(k in comb for k in ESG_KEYWORDS):
            score -= 0.4

    return score


# Load
print("Loading workbook...")
df = pd.read_excel(INPUT_PATH)
original_shape = df.shape
print(f"Loaded {original_shape[0]} rows, {original_shape[1]} columns")

# Keep only existing metadata columns in original order if present
existing_columns = [c for c in REQUIRED_COLUMNS if c in df.columns]
missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")

# Preserve all existing columns and keep them in original order; add missing columns as empty if needed
for c in REQUIRED_COLUMNS:
    if c not in df.columns:
        df[c] = np.nan

# Ensure scope exists; fill empty with Other if needed
if "scope" not in df.columns:
    df["scope"] = "Other"
df["scope"] = df["scope"].fillna("Other")

# Clean 1: remove obvious low-quality rows
before_len = len(df)

df = df.copy()

# Remove rows with empty or unusable text
mask_meaningful = df["text"].apply(is_meaningful_text)
# Keep scope-related short text if it contains emission-related keywords
for idx, row in df.iterrows():
    txt = normalize_text(row.get("text", ""))
    if not mask_meaningful.loc[idx] and any(k in txt for k in ["scope 1", "scope 2", "scope 3", "emission", "emissions", "ghg", "carbon", "electricity", "diesel", "fuel", "travel", "waste", "water"]):
        mask_meaningful.loc[idx] = True

df = df[mask_meaningful].copy()

# Remove exact duplicate normalized text (keep highest quality row)
normalized_text = df["text"].apply(normalize_text)
df["_norm_text"] = normalized_text

df = df.sort_values(["_norm_text", "scope", "semantic_score"], ascending=[True, True, False], na_position="last")
# keep first occurrence per normalized text

df = df.drop_duplicates(subset=["_norm_text"], keep="first")

# Remove duplicate block_id (keep highest quality row)
df["block_id"] = df["block_id"].fillna("")
df["_block_id_norm"] = df["block_id"].astype(str).str.strip()
df = df.sort_values(["_block_id_norm", "scope", "semantic_score"], ascending=[True, True, False], na_position="last")
df = df.drop_duplicates(subset=["_block_id_norm"], keep="first")

# Remove rows with empty block_id or weird block_id? Keep if meaningful text is present; no need.

# Remove rows with empty text again after duplicates cleanup
mask_meaningful2 = df["text"].apply(is_meaningful_text)
df = df[mask_meaningful2].copy()

removed_rows = before_len - len(df)
duplicate_removed = before_len - len(df) - removed_rows

# Compute quality score
print("Scoring rows for selection...")
df["quality_score"] = df.apply(quality_score, axis=1)

# Rebalance classes
class_targets = {"Scope 1": 1000, "Scope 2": 1000, "Scope 3": 1700, "Other": 1700}

balanced_rows = []
summary = {"before": Counter(df["scope"].fillna("Other")), "after": Counter(), "removed": removed_rows, "duplicate_removed": duplicate_removed, "downsampled": 0, "oversampled": 0}

# Downsample or keep each class
for label in ["Scope 1", "Scope 2", "Scope 3", "Other"]:
    sub = df[df["scope"] == label].copy()
    if len(sub) == 0:
        continue

    # Sort for quality; for Other and Scope 3, downsample; for Scope 1 and 2, keep or oversample.
    sub = sub.sort_values(["quality_score", "semantic_score"], ascending=[False, False], na_position="last")

    if label in ["Other", "Scope 3"]:
        target = class_targets[label]
        if len(sub) > target:
            selected = sub.head(target).copy()
            summary["downsampled"] += len(sub) - len(selected)
        else:
            selected = sub.copy()
    else:
        target = class_targets[label]
        if len(sub) >= target:
            selected = sub.head(target).copy()
        else:
            selected = sub.copy()
            # Oversample with replacement using weighted sampling
            need = target - len(selected)
            weights = np.maximum(selected["quality_score"].fillna(0).to_numpy(), 0.01)
            weights = weights / weights.sum()
            sampled = np.random.choice(selected.index, size=need, replace=True, p=weights)
            sampled_rows = selected.loc[sampled].copy()
            selected = pd.concat([selected, sampled_rows], ignore_index=True)
            summary["oversampled"] += need

    balanced_rows.append(selected)
    summary["after"][label] = len(selected)

balanced_df = pd.concat(balanced_rows, ignore_index=True)

# Keep only metadata columns in original order and ensure no accidental temp columns remain
for c in ["_norm_text", "_block_id_norm"]:
    if c in balanced_df.columns:
        balanced_df = balanced_df.drop(columns=[c])

# Reorder columns to preserve metadata columns as originally available
ordered = [c for c in df.columns if c in balanced_df.columns]
balanced_df = balanced_df[ordered]

# Write outputs
balanced_df.to_excel(OUTPUT_XLSX, index=False)
balanced_df.to_csv(OUTPUT_CSV, index=False)

# Print report
print("\n=== Balancing report ===")
print("Before")
print(dict(Counter(df["scope"].fillna("Other"))))
print("\nAfter")
print(dict(Counter(balanced_df["scope"].fillna("Other"))))
print(f"\nRows removed during cleaning: {removed_rows}")
print(f"Duplicate-like rows removed: {summary['duplicate_removed']}")
print(f"Rows downsampled: {summary['downsampled']}")
print(f"Rows oversampled: {summary['oversampled']}")
print(f"\nSaved to: {OUTPUT_XLSX}")
print(f"Saved to: {OUTPUT_CSV}")

# Write reason.md
reason_text = f"""# Reasoning for balancing the scope dataset

- The original workbook was read first and inspected for schema, class distribution, and sample rows before any transformation.
- The balancing process followed the requested priority order:
  1. Remove obvious low-quality rows: duplicate normalized text, duplicate block_id, very short or meaningless OCR-like text, and clearly irrelevant header/footer/copyright-like content.
  2. Downsample Other by keeping high-signal ESG-related examples rather than random sampling. Rows with strong climate/energy/sustainability vocabulary and higher semantic similarity were prioritized.
  3. Downsample Scope 3 only lightly because it already sits near the target size and should remain diverse.
  4. For Scope 1 and Scope 2, the dataset was kept as-is first and then oversampled with replacement from existing high-quality rows only. No new text was created, no paraphrase was used, and no synthetic data was generated.
- The output files preserve the existing metadata columns and only adjust the row selection to better support ClimateBERT fine-tuning.

## Before / After counts
- Before: Scope 1={Counter(df['scope'].fillna('Other'))['Scope 1']}, Scope 2={Counter(df['scope'].fillna('Other'))['Scope 2']}, Scope 3={Counter(df['scope'].fillna('Other'))['Scope 3']}, Other={Counter(df['scope'].fillna('Other'))['Other']}
- After: Scope 1={Counter(balanced_df['scope'].fillna('Other'))['Scope 1']}, Scope 2={Counter(balanced_df['scope'].fillna('Other'))['Scope 2']}, Scope 3={Counter(balanced_df['scope'].fillna('Other'))['Scope 3']}, Other={Counter(balanced_df['scope'].fillna('Other'))['Other']}

## Summary
- Rows removed during cleaning: {removed_rows}
- Duplicate-like rows removed: {summary['duplicate_removed']}
- Rows downsampled: {summary['downsampled']}
- Rows oversampled: {summary['oversampled']}
"""
with open(REASON_PATH, "w", encoding="utf-8") as f:
    f.write(reason_text)

print(f"Wrote explanation note to {REASON_PATH}")
