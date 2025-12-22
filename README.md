# OGC Evals

## Overview

OGC Evals is a robust evaluation framework designed to benchmark Large Language Models (LLMs) on "citizen queries"—open-ended, prompt-response interactions typical of government or civic information seeking. 

This project implements a **FActScore-style** evaluation pipeline, enhancing the original methodology by integrating insights from **SAFE** (Search-Augmented Factuality Evaluators) and **VeriScore**. The goal is to provide granular metrics on model performance, specifically focusing on factuality and the ability to answer questions without unnecessary hallucination or improper abstention.

## Methodology

The evaluation pipeline consists of three distinct stages:

### 1. Abstention Detection
Before checking for factual accuracy, the system determines if the model actually attempted to answer the prompt.
- **Goal**: Distinguish between "hallucinations" (incorrect answers) and "refusals" (e.g., "I cannot answer that").
- **Metric**: Abstention Rate.
- **Implementation**: `ogc_eval/abstention.py` uses an LLM classifier to detect refusal patterns.

### 2. Atomic Fact Generation (AFG)
Responses (both Ground Truth and Generated) are decomposed into individual, self-contained "atomic facts".
- **Approach**: Adapted from the **SAFE** architecture but modified to produce **declarative statements** rather than search queries.
- **Mechanism**: Utilizes a "demon" retrieval system (located in `ogc_eval/demos`) to dynamically inject relevant few-shot examples into the system prompt based on the input text. This ensures the atomization is context-aware and robust.
- **Implementation**: `ogc_eval/afg.py`.

### 3. Automatic Fact Verification (AFV)
An "LLM-as-a-judge" compares the atomic facts extracted from the model's response against the atomic facts from the ground truth.
- **Metric**: **Accuracy@k**. We measure the number of claims in the generated response ($A$) that are entailed by the ground truth claims ($C$), capped at the number of claims in the ground truth ($k$). This prevents models from gaming the metric by generating excessive verifiable but irrelevant facts.
- **Implementation**: `ogc_eval/afv.py`.

## Project Structure

```
OGC_evals/
├── ogc_eval/
│   ├── abstention.py   # Module for detecting model refusals
│   ├── afg.py          # Atomic Fact Generator (with SAFE-style demons)
│   ├── afv.py          # Automatic Fact Verifier (Entailment checks)
│   ├── demos/          # Few-shot examples (demons) for AFG
│   ├── main.py         # Orchestrator script for running evaluations
│   └── model.py        # LLM Wrapper (currently Mock/CPU-based)
├── datasetsample.csv   # Sample input dataset (prompts + expected responses)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1. **Clone the repository** and navigate to the root directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Spacy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

To run the evaluation suite on the provided sample dataset:

```bash
python -m ogc_eval.main
```

This will:
1. Load `datasetsample.csv`.
2. Run the full pipeline (Abstention -> AFG -> AFV).
3. Output the detailed metrics to `eval_results.csv`.

## Current Status

*   **Architecture**: The core modular architecture is complete and self-contained within the `ogc_eval` package. Dependencies on external `long-form-factuality` folders have been removed.
*   **Model Integration**: The system is currently running on a **Mock LLM Wrapper** (`ogc_eval/model.py`) to facilitate development without GPU access. 
    *   *Action Required*: To perform real evaluations, update `ogc_eval/model.py` to initialize a real Transformer model (e.g., Llama-3, Mistral) or connect to an API endpoint.
*   **Dataset**: currently mocks "generated responses" by copying ground truth data for pipeline verification.

## References

*   **FActScore**: Min et al. (2023). [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251).
*   **SAFE**: Wei et al. (2024). [Long-form Factuality in Large Language Models](https://arxiv.org/abs/2403.18802).
*   **VeriScore**: Song, Kim & Iyyer (2024). [VeriScore: Evaluating the Factuality of Verifiable Claims in Long-Form Text Generation](https://arxiv.org/abs/2406.19276).
*   **OpenFActScore**: Lage & Ostermann (2025). [OpenFActScore: Open-Source Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2507.05965).
