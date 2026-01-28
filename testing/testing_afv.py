# testing file
# to test components iteratively

# current test: AFV (Fact Verification)
# testing with instruct model Gemma 3 270m on CPU
# using a dataset sample where generated responses are just a copy of expected responses
# requires auth with huggingface

# 21/01/26

import pandas as pd
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator
from ogc_eval.abstention import AbstentionDetector
from ogc_eval.afv import FactVerifier

# Constants
TEST_SAMPLE_SIZE = 5 # Small sample to save time
INPUT_FILE = "public_set_1.csv"

def run_test():
    print(f"Loading first {TEST_SAMPLE_SIZE} rows from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE).head(TEST_SAMPLE_SIZE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    if 'generated_response' not in df.columns:
        print("Error: 'generated_response' column missing in CSV. Please add it first.")
        return
   
    llm = LLMWrapper(
        model_name="google/gemma-3-270m-it",
        device="cpu",
        mock=False 
    )

    abstention_detector = AbstentionDetector(device="cpu") # Use CPU for test

    afg = AtomicFactGenerator(llm)
    verifier = FactVerifier(llm)

    for index, row in df.iterrows():
        prompt = row['prompt']
        ground_truth = row['response']
        generated_response = row['generated_response'] 
        
        print(f"Row {index + 1}:") # type: ignore
        print(f"  Prompt: {prompt[:50]}...")
        
        # 1. Abstention Detection
        judgement = abstention_detector.is_abstention(generated_response)
        print(f"Abstention check: {judgement}")

        # 2. Atomic Fact Generation (Ground Truth)
        # Ideally pre-computed, but for this test script we run it live
        gt_facts, _ = afg.run(ground_truth)
        print(f"Ground Truth -> Found {len(gt_facts)} facts: {json.dumps(gt_facts[:2])}...")

        # 3. Atomic Fact Generation (Generated Response)
        gen_facts, _ = afg.run(generated_response)
        print(f"Generated Response -> Found {len(gen_facts)} claims: {json.dumps(gen_facts[:2])}...")
        
        # 4. Automatic Fact Verification
        score, supported_count = verifier.verify(gen_facts, gt_facts)
        
        print(f"Accuracy Score: {score:.2f} ({supported_count}/{len(gt_facts)} supported)")
        print("-" * 40)

if __name__ == "__main__":
    run_test()
