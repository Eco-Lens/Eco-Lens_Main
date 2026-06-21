# Eco-Lens

> Automated Carbon Footprint Quantification & ESG Analysis using Explainable AI

## Overview
![Overview](img/Module%201.jpg)

Eco-Lens is an AI-powered system that automates the analysis of ESG (Environmental, Social, and Governance) reports. It reads complex ESG documents in PDF or digitized formats, extracts carbon emission data, classifies emissions by GHG Protocol scopes (Scope 1, 2, 3), and cross-references results against international standards via RAG to support audits and investment decisions.

## Problem Statement

Modern ESG reports are structurally complex — mixing narrative text, data tables, charts, footnotes, and domain-specific terminology across hundreds of pages. Manual extraction and verification are:

- **Time-consuming** and labor-intensive
- **Difficult to standardize** across different reporting formats
- **Prone to misclassification** of emission scopes (Scope 1 vs. 2 vs. 3)
- **Hard to audit** — black-box AI predictions lack traceability

## Target Users

| User Group | Primary Needs |
|---|---|
| **Business Leaders** | Monitor emission trends, track Net Zero progress, support strategic decisions (e.g., supplier change, solar investment), forecast future carbon costs, ensure compliance |
| **Auditors / Regulators / Third Parties** | Verify ESG authenticity, evaluate environmental compliance |
| **Investors** | Assess long-term sustainability risk, evaluate governance transparency, identify supply chain carbon exposure, screen ESG portfolio eligibility |

## Architecture

### Module 1 — Multimodal ESG Report Understanding (Core)

The core pipeline converts raw ESG documents into structured, classified, and standards-validated emission data.

```
PDF ESG Report
      ↓
[1] PaddleOCR              — Text extraction
      ↓
[2] LayoutLMv3             — Layout understanding (text + position + image)
      ↓
[3] Microsoft Table Transformer — Table structure parsing (row/column relations)
      ↓
[4] ClimateBERT            — Semantic mapping to Scope 1 / 2 / 3
      ↓
[5] BGE-large + RAG        — Standards retrieval (GHG Protocol, GRI Standards)
      ↓
Structured ESG Data
```

#### Detailed Components

| Step | Model | Role |
|---|---|---|
| OCR | **PaddleOCR** | Converts PDF/scanned images to machine-readable text with strong table support |
| Layout Understanding | **LayoutLMv3** | Understands document structure — distinguishes headers, tables, paragraphs, footers, and ESG metric blocks |
| Table Understanding | **Microsoft Table Transformer (TATR/DETR-based)** | Parses complex ESG tables including merged cells, multi-row headers, and nested structures |
| Semantic Mapping | **ClimateBERT** (fine-tuned) | Classifies emission entries into Scope 1, 2, or 3 based on climate-domain semantic understanding |
| Standards Retrieval | **BGE-large** + RAG | Retrieves relevant clauses from GHG Protocol and GRI Standards to validate and augment classification |

#### Expected Output (Module 1)

```json
{
  "company": "ABC Corp",
  "year": 2025,
  "scope": "Scope 2",
  "emission_value": 15234,
  "unit": "tCO2e",
  "source_page": 48,
  "retrieved_standard": "GHG Protocol Scope 2 Guidance"
}
```

## Data Strategy

| Source | Details |
|---|---|
| **Primary Dataset** | 300+ sustainability reports from Vietnamese enterprises |
| **Augmentation** | Hugging Face NLP datasets for climate-text classification, financial sentiment |
| **Standards Knowledge Base** | GHG Protocol & GRI Standards (permanently connected via RAG) |
| **Legal Context** | Climate Change Laws of the World dataset |
| **Train/Test Split** | 80% training / 20% testing |

The primary dataset uses Vietnamese enterprise reports to ensure relevance to local regulatory context (e.g., Decree 06/2022/NĐ-CP on greenhouse gas emission reduction). Data augmentation and global standards integration mitigate bias from the limited domestic dataset size.

## Implementation Timeline

| Phase | Duration | Focus |
|---|---|---|
| **Phase 1** | Weeks 1–4 | Data collection, cleaning, and labeling; PaddleOCR text extraction; LayoutLMv3 pre-training on ESG document layout |
| **Phase 2** | Weeks 5–8 | Fine-tune Microsoft Table Transformer for emission table parsing; handle merged cells and multi-row headers |
| **Phase 3** | Weeks 9–12 | Integrate ClimateBERT for semantic scope classification; deploy BGE-large RAG pipeline with GHG/GRI standards |
| **Phase 4** | Weeks 13–16 | Full pipeline integration and rigorous testing on real ESG reports |

**Total: 16 weeks** for Module 1 completion.

## Expected Outcomes

1. An automated AI pipeline that reads and understands multimodal ESG reports
2. A cleaned dataset of extracted emission figures, normalized to international standards
3. An intelligent classifier automatically assigning emissions to Scope 1, 2, and 3
4. Structured ESG data output with full provenance (source page, unit, referenced standard)
5. A prototype that reduces manual ESG processing workload and increases data verifiability for enterprises and auditors

## Technology Stack

| Component | Technology |
|---|---|
| OCR | PaddleOCR |
| Layout Understanding | LayoutLMv3 |
| Table Parsing | Microsoft Table Transformer |
| Scope Classification | ClimateBERT |
| Standards Retrieval (RAG) | BGE-large Embeddings |
| Vector Database | FAISS / ChromaDB |
| Framework | PyTorch, Hugging Face Transformers |
=======

