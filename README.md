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
│   ├── data_loader.py  # Dataset loading and validation (CSV/JSONL)
│   ├── demos/          # Few-shot examples (demons) for AFG
│   ├── main.py         # Orchestrator script for running evaluations
│   └── model.py        # LLM Wrapper (Mock, HuggingFace, OpenAI API)
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

To run the evaluation suite on the provided sample dataset (defaults to **Mock Mode**):

```bash
python -m ogc_eval.main
```

To use a real model, instantiate `OGCEvaluator` in a script with `mock=False` and provide an API key or HuggingFace model path:

```python
from ogc_eval.main import OGCEvaluator

# OpenAI
evaluator = OGCEvaluator(model_name="gpt-4", api_key="sk-...", mock=False)

# HuggingFace (Local GPU)
evaluator = OGCEvaluator(model_name="meta-llama/Llama-3-8b", device="cuda", mock=False)

evaluator.evaluate_dataset("datasetsample.csv")
```

## Current Status

*   **Architecture**: The core modular architecture is complete and self-contained within the `ogc_eval` package. Dependencies on external `long-form-factuality` folders have been removed.
*   **Model Integration**: `ogc_eval/model.py` is fully implemented to support:
    *   **Mock Mode**: For testing logic without resources.
    *   **OpenAI API**: Auto-detects `OPENAI_API_KEY` or accepts explicit key.
    *   **HuggingFace Transformers**: Loads local models (supports `device_map="auto"`).
*   **Dataset**: Pipeline currently supports loading CSV/JSONL. If 'generated_response' is missing, it mocks it by copying ground truth data for pipeline verification.

## Notes & Idiosyncrasies

*   **Mock Mode Default**: The system defaults to `mock=True` to allow running the pipeline without API keys or GPUs. In this mode, the LLM generates static placeholder text (e.g., "- Mock atomic fact 1"). Ensure you set `mock=False` for actual evaluations.
*   **Abstention Detection Model**: The abstention module uses a specialized local model (`LibrAI/longformer-action-ro`). This model is approximately **600MB** and will be downloaded upon first use. It is designed to run efficiently on a CPU, so it remains active even when the main evaluator is in "Mock Mode".
*   **Data Fallback**: If the input dataset (CSV/JSONL) is missing the `generated_response` column (i.e., you are testing the pipeline structure rather than a model's output), the system will automatically copy the ground truth `response` into `generated_response`. This facilitates pipeline testing but means "perfect" scores in this scenario are an artifact of this fallback.
*   **Spacy Dependency**: The Atomic Fact Generator (AFG) strictly requires the `en_core_web_sm` Spacy model for sentence splitting. Ensure this is downloaded as per the installation instructions.

## References

*   **FActScore**: Min et al. (2023). [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251).
*   **SAFE**: Wei et al. (2024). [Long-form Factuality in Large Language Models](https://arxiv.org/abs/2403.18802).
*   **VeriScore**: Song, Kim & Iyyer (2024). [VeriScore: Evaluating the Factuality of Verifiable Claims in Long-Form Text Generation](https://arxiv.org/abs/2406.19276).
*   **OpenFActScore**: Lage & Ostermann (2025). [OpenFActScore: Open-Source Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2507.05965).
