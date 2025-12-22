# OGC Evals: Roadmap & Next Steps

This document outlines the immediate and short-term tasks required to move `OGC_evals` from a mock architecture to a production-ready evaluation suite.

## 1. Data Pipeline Refinement (User Lead)
*   **Update Dataset:** Replace `datasetsample.csv` with the comprehensive dataset.
*   **Schema Standardization:** Ensure the dataset clearly distinguishes between:
    *   `prompt`: The citizen query.
    *   `response` (Ground Truth): The verified correct answer.
    *   `generated_response`: The output from the LLM being tested (if running evaluation offline).
*   **Pipeline Integration:** Refine how data flows into `ogc_eval`.
    *   *Suggestion:* We should implement a `DatasetRegistry` or `DataLoader` class within `ogc_eval` (inspired by the snippets you provided) to handle data loading more robustly than the current `pd.read_csv` implementation.

## 2. Model Response Pipelines (User Lead)
*   **Generation Source:** Determine how the model under test generates responses.
    *   *Option A (Offline/Batch):* You provide a CSV that already contains a `generated_response` column.
    *   *Option B (Live/Online):* We configure `ogc_eval` to query the model in real-time during the evaluation loop.
*   **Evaluator Model Config:** Define which model will serve as the "Judge" for AFG and AFV (e.g., GPT-4o, Llama-3-70B). The quality of the judge is critical for the FActScore metric.

## 3. Architecture Implementation (Agent Lead)
*   **De-Mocking `model.py`:** Update the `LLMWrapper` to connect to real inference endpoints (HuggingFace, OpenAI, vLLM, etc.) based on your preference.
*   **Prompt Engineering:**
    *   Review `abstention.py` prompts to ensure they catch subtle refusals.
    *   Review `afv.py` entailment prompts to ensure accuracy in verifying facts.
*   **Aggregator Logic:** Implement logic to calculate the final aggregate scores (e.g., Average FActScore, Abstention Rate, Accuracy@k) rather than just row-by-row results.

## 4. Output Design (Collaborative)
*   **Granularity:** Define what comes out of the pipe.
    *   *Current:* A CSV with atomic facts and scores per row.
    *   *Needed:* A summary report (JSON/Markdown) summarizing the model's behavior (e.g., "The model abstained on 15% of questions," "Factuality score dropping on long-context queries").

## Checklist for Next Session
- [ ] User: Provide the full dataset (or a larger, real slice of it).
- [ ] User: Specify the LLM backend/API details for the Evaluator model.
- [ ] Agent: Implement `DatasetRegistry` / Data Loading logic.
- [ ] Agent: Implement real `generate()` and `classify()` methods in `model.py`.
