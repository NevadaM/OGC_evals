# testing file
# to test components iteratively

# current test: AFV (Fact Verification)
# testing with instruct model Gemma 3 270m on CPU
# requires auth with huggingface

# 01/01/26

import pandas as pd
import sys
import os
import json

# Ensure we can import ogc_eval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator
from ogc_eval.abstention import AbstentionDetector
from ogc_eval.afv import FactVerifier

# Constants
TEST_SAMPLE_SIZE = 3 # Small sample to save time
INPUT_FILE = "datasetsample.csv"

def run_test():
    # Load Data
    print(f"Loading first {TEST_SAMPLE_SIZE} rows from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE).head(TEST_SAMPLE_SIZE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Check for generated_response
    if 'generated_response' not in df.columns:
        print("Error: 'generated_response' column missing in CSV. Please add it first.")
        return

    # Initialize Model (Small CPU model for testing)
    print("Initializing LLM (Gemma 270m)...")
    # Warning: This requires HF Token if the model is gated, but Gemma 270m might be open or cached.
    # The user mentioned "I'll run things in a separate instance... don't worry about it".
    try:
        llm = LLMWrapper(
            model_name="google/gemma-3-270m-it",
            device="cpu",
            mock=False 
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        print("Falling back to Mock Mode for test structure verification...")
        llm = LLMWrapper(mock=True)

    # Initialize Components
    print("Initializing Components...")
    try:
        abstention_detector = AbstentionDetector(device="cpu") # Use CPU for test
    except Exception as e:
        print(f"Failed to init AbstentionDetector (maybe model download issue): {e}")
        return

    afg = AtomicFactGenerator(llm)
    verifier = FactVerifier(llm)

    # Run Pipeline
    print("\n--- Starting Pipeline Test ---\n")

    for index, row in df.iterrows():
        prompt = row['prompt']
        ground_truth = row['response']
        generated_response = row['generated_response'] 
        
        print(f"Row {index + 1}:")
        print(f"  Prompt: {prompt[:50]}...")
        
        # 1. Abstention Detection
        try:
            label_id, label_desc, score = abstention_detector.is_abstention(generated_response)
            print(f"  [Abstention] Label: {label_desc} ({score:.4f})")
            
            if label_id in [0, 4]: # If abstained (0=Refusal, 4=Incapable/Other depending on map)
                print("  -> Model Abstained. Skipping AFG/AFV.")
                continue
        except Exception as e:
            print(f"  [Abstention] Error: {e}")

        # 2. Atomic Fact Generation (Ground Truth)
        # Ideally pre-computed, but for this test script we run it live
        print("  [AFG] Atomizing Ground Truth...")
        gt_facts, _ = afg.run(ground_truth)
        print(f"    -> Found {len(gt_facts)} facts: {json.dumps(gt_facts[:2])}...")

        # 3. Atomic Fact Generation (Generated Response)
        print("  [AFG] Atomizing Generated Response...")
        gen_facts, _ = afg.run(generated_response)
        print(f"    -> Found {len(gen_facts)} facts: {json.dumps(gen_facts[:2])}...")
        
        # 4. Automatic Fact Verification
        print("  [AFV] Verifying Facts...")
        score, supported_count = verifier.verify(gen_facts, gt_facts)
        
        print(f"  [Result] Accuracy Score: {score:.2f} ({supported_count}/{len(gt_facts)} supported)")
        print("-" * 40)

    print("\nTest Complete.")

if __name__ == "__main__":
    run_test()
