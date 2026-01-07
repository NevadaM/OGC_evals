# OGC Evals

## Overview

OGC Evals is a robust evaluation framework designed to benchmark Large Language Models (LLMs) on "citizen queries"—open-ended, prompt-response interactions typical of government or civic information seeking. 

It implements a **FActScore-style** evaluation pipeline, enhancing the original methodology by integrating insights from **SAFE** (Search-Augmented Factuality Evaluators) and **VeriScore**.

## The Standard Workflow

The OGC Evals architecture is designed to save time and compute by splitting the process into two phases: **Reference Creation** and **Model Benchmarking**.

### Step 1: Initialize the Reference Dataset (`prepare`)
**Frequency:** Once per dataset.  
**Purpose:** The "Ground Truth" answers in your dataset must be decomposed into Atomic Facts before they can be used for evaluation. Since this requires LLM calls (using the AFG module), we pre-compute and save these facts so they don't need to be re-generated for every model you test.

```bash
# Example: Prepare the sample dataset (Mock mode)
python -m ogc_eval.main prepare --input datasetsample.csv --output datasetsample_reference.csv --mock
```

**Output:** A **Reference Dataset** (e.g., `datasetsample_reference.csv`) containing a `response_facts` column. This file is now the "source of truth" for all future evaluations.

### Step 2: Evaluate Model Outputs (`evaluate`)
**Frequency:** Many times (Once per model or experiment).  
**Purpose:** This stage takes your model's generated responses, extracts facts from them, and compares them against the **Reference Dataset** created in Step 1.

```bash
# Example: Evaluate a model using the Reference Dataset
python -m ogc_eval.main evaluate --input datasetsample_reference.csv --mock
```

> **Note:** If you run `evaluate` on a raw dataset (without running `prepare` first), the system will calculate Ground Truth facts on-the-fly. This works but is significantly slower and wastes tokens if you plan to evaluate multiple models on the same data.

**Pipeline Details:**
1.  **Abstention Check:** Did the model refuse to answer? (Local PLM classifier)
2.  **Fact Generation:** Atomize the *model's* response.
3.  **Fact Verification:** Compare Model Facts vs. Reference Facts (LLM Judge).

## Outputs
- **Results CSV:** `ogc_eval_results_{timestamp}.csv` containing granular, row-by-row scores.
- **Summary Report:** `ogc_eval_results_{timestamp}_summary.txt` containing aggregate metrics (Mean Accuracy, Abstention Rate, etc.).
- **Logs:** `logs/ogc_eval_{timestamp}.log`.

## Project Architecture

```
OGC_evals/
├── ogc_eval/
│   ├── main.py         # CLI Entry point (prepare & evaluate commands)
│   ├── abstention.py   # PLM-based refusal detection
│   ├── afg.py          # Atomic Fact Generator
│   ├── afv.py          # Fact Verifier (Entailment)
│   ├── data_loader.py  # Handles CSV/JSONL & Batching
│   ├── result_writer.py# Generates CSVs and Summary Reports
│   ├── logger.py       # Centralized logging
│   ├── model.py        # Wrapper for OpenAI/HF/Mock models
│   ├── demos/          # Few-shot examples
│   └── prompts/        # System prompts
├── logs/               # Execution logs
└── requirements.txt    # Python dependencies
```

## Methodology

### 1. Abstention Detection
Uses a **PLM-based classifier** (`LibrAI/longformer-action-ro`) to detect if a model response is a refusal or disclaimer. This prevents "I cannot answer" from being halluncinated as a factual claim.

### 2. Atomic Fact Generation (AFG)
Decomposes long-form text into self-contained "atomic facts".
- **Ground Truth**: Pre-computed during the `prepare` stage.
- **Generated Response**: Computed dynamically during `evaluate`.
- **Mechanism**: Uses `spacy` for sentence splitting and an LLM with few-shot "demons" to extract facts.

### 3. Automatic Fact Verification (AFV)
An "LLM-as-a-judge" compares the generated facts against the ground truth facts to calculate **Accuracy@k**.

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Spacy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Hardware Requirements

Resource usage depends heavily on whether you are using **OpenAI API** or **Local Models (HuggingFace)**.

### Disk Space
- **Abstention Classifier**: ~600MB (Automatically downloaded to HuggingFace cache on first use).
- **Spacy Model**: ~15MB.
- **Evaluation Models**: Variable (e.g., Llama-3-8B is ~15GB).

### RAM / Memory
- **Base Requirement**: 8GB RAM is recommended for data processing and running the Abstention Classifier pipeline comfortably.
- **Large Datasets**: For datasets with >10,000 rows, ensure sufficient RAM for Pandas operations.

### GPU / vRAM
- **Abstention Detection**: Can run on CPU, but uses ~1GB vRAM if `device="cuda"` is specified.
- **Local LLMs**:
    - **8B Models (e.g., Llama-3)**: Require ~16GB-24GB vRAM (depending on quantization) for the `prepare` and `evaluate` stages if run locally.
    - **70B Models**: Require multiple GPUs or high-memory A100/H100 instances.
- **API-only usage**: If using `mock=True` or OpenAI models, the GPU requirement is effectively zero.
