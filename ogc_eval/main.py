# main
## not checked

import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional

from .model import LLMWrapper
from .afg import AtomicFactGenerator
from .abstention import AbstentionDetector
from .afv import FactVerifier
from .data_loader import DataLoader

class OGCEvaluator:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu", api_key: Optional[str] = None, mock: bool = False):
        self.model = LLMWrapper(model_name, device, api_key=api_key, mock=mock)
        self.afg = AtomicFactGenerator(self.model)
        self.abstention_detector = AbstentionDetector(self.model)
        self.verifier = FactVerifier(self.model)

    def evaluate_dataset(self, csv_path: str, output_path: str = "eval_results.csv"):
        loader = DataLoader(csv_path)
        df = loader.load()
        
        # Mocking 'generated_response' if not present
        if 'generated_response' not in df.columns:
            print("Column 'generated_response' not found. Creating mock responses for testing (copying ground truth).")
            # Mock: use the ground truth but truncate it or modify it slightly
            df['generated_response'] = df['response'].apply(lambda x: x) 

        results = []
        
        print("Starting evaluation...")
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            prompt = row['prompt']
            ground_truth = row['response']
            generated = row['generated_response']
            
            # 1. Abstention Detection
            is_abstained = self.abstention_detector.is_abstention(prompt, generated)
            
            if is_abstained:
                results.append({
                    "prompt": prompt,
                    "is_abstained": True,
                    "afg_k_gt": 0,
                    "afg_k_gen": 0,
                    "score": 0.0
                })
                continue

            # 2. Atomic Fact Generation (AFG)
            # Ground Truth Claims
            gt_claims, k_gt = self.afg.run(ground_truth)
            
            # Generated Claims
            gen_claims, k_gen = self.afg.run(generated)
            
            # 3. Automatic Fact Verification (AFV)
            score, supported_count = self.verifier.verify(gen_claims, gt_claims)
            
            results.append({
                "prompt": prompt,
                "is_abstained": False,
                "afg_k_gt": k_gt,
                "afg_k_gen": k_gen,
                "supported_claims": supported_count,
                "score": score,
                "gt_claims": str(gt_claims),
                "gen_claims": str(gen_claims)
            })

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Results saved to {output_path}")
        return result_df

if __name__ == "__main__":
    # Default to mock mode for safe execution without GPU/API keys
    evaluator = OGCEvaluator(mock=True)
    evaluator.evaluate_dataset("datasetsample.csv")
