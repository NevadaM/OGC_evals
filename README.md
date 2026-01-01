# OGC Evals

## Overview

OGC Evals is a robust evaluation framework designed to benchmark Large Language Models (LLMs) on "citizen queries"—open-ended, prompt-response interactions typical of government or civic information seeking. 

It implements a **FActScore-style** evaluation pipeline, enhancing the original methodology by integrating insights from **SAFE** (Search-Augmented Factuality Evaluators) and **VeriScore**.

## Workflow & Usage

The evaluation process is split into two distinct stages to maximize efficiency and robustness.

### 1. Preparation (`prepare`)
**Goal:** Pre-compute Atomic Facts for the Ground Truth responses.
**Why:** Running Atomic Fact Generation (AFG) on the ground truth for every evaluation run is redundant. This step runs it once and saves a "Golden Dataset".

```bash
# Basic usage (Mock mode)
python -m ogc_eval.main prepare --input datasetsample.csv --mock --output datasetsample_prepped.csv

# With a real model (OpenAI)
python -m ogc_eval.main prepare --input datasetsample.csv --model gpt-4 --api_key sk-...
```

**Output:** A CSV file (e.g., `datasetsample_prepped.csv`) containing a new `response_facts` column (JSON-serialized list of facts).

### 2. Evaluation (`evaluate`)
**Goal:** Evaluate a model's generated responses against the Golden Dataset.
**Pipeline:**
1.  **Abstention Detection:** Checks if the model refused to answer (using `LibrAI/longformer-action-ro`).
2.  **AFG (Gen):** Decomposes the model's generated response into atomic facts.
3.  **AFV:** Verifies these facts against the pre-computed Ground Truth facts using an LLM Judge.

```bash
# Run evaluation on the prepared dataset
python -m ogc_eval.main evaluate --input datasetsample_prepped.csv --mock

# With a local HuggingFace model
python -m ogc_eval.main evaluate --input datasetsample_prepped.csv --model meta-llama/Llama-3-8b --device cuda
```

**Output:**
- **Results CSV:** `ogc_eval_results_{timestamp}.csv` (Detailed row-by-row metrics).
- **Summary Report:** `ogc_eval_results_{timestamp}_summary.txt` (Aggregated statistics).
- **Logs:** `logs/ogc_eval_{timestamp}.log` (Execution trace).

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
