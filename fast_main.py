import argparse
import os
import json
import pandas as pd
import time
import random
import threading
import glob
import logging
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. IMPORT LOGIC ---
from ogc_eval.data_loader import DataLoader
from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator
from ogc_eval.abstention import AbstentionDetector
from ogc_eval.afv import FactVerifier
from ogc_eval.result_writer import ResultWriter
from ogc_eval.logger import setup_logger

# --- 2. SILENT & ROBUST API CLIENT ---
import litellm
from litellm import completion as original_completion

litellm.suppress_debug_info = True
litellm.drop_params = True
# Nuke all logging from orbit
for logger_name in ["litellm", "httpx", "httpcore"]:
    l = logging.getLogger(logger_name)
    l.setLevel(logging.CRITICAL)
    l.propagate = False

def robust_completion(*args, **kwargs):
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 120.0 

    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            return original_completion(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Catch everything that looks like a temp failure
            if any(x in error_str for x in ["rate limit", "429", "503", "service unavailable", "overloaded", "timeout", "timed out"]):
                attempt += 1
                wait = 10 + (attempt * 5) + random.uniform(1, 5)
                time.sleep(wait)
                continue
            raise e
    raise Exception("Max Retries Exceeded")

litellm.completion = robust_completion

import ogc_eval.model
ogc_eval.model.completion = robust_completion

# --- 3. NEW: BATCH FACT VERIFIER (The Speed Fix) ---
class BatchFactVerifier(FactVerifier):
    """
    Verifies ALL claims in a single API call instead of looping.
    Reduces API usage by ~90%.
    """
    def verify(self, hypothesis_claims, reference_claims):
        if not hypothesis_claims:
            return 0.0, 0
            
        if not reference_claims: # If GT is empty, nothing is supported
            return 0.0, 0

        # Construct a Single Prompt
        ref_text = "\n".join([f"- {c}" for c in reference_claims])
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(hypothesis_claims)])
        
        prompt = f"""You are an objective fact-checker.
        
Reference Facts:
{ref_text}

Claims to Verify:
{claims_text}

For EACH claim numbered 1 to {len(hypothesis_claims)}, determine if it is supported by the Reference Facts.
Output ONLY a JSON list of strings, where each entry is "YES" (supported) or "NO" (not supported).
Do not provide reasoning, preamble, or any other text.
JSON Output: ["""
        try:
            # 1. Generate with JSON mode and a tight token limit
            response = self.model.generate(
                prompt, 
                max_new_tokens=150, 
                response_format={ "type": "json_object" } 
            ).strip()
            
            # 2. Parse the JSON string into a Python object
            data = json.loads(response)
            
            # 3. Extract the list (handles both raw list or wrapped dict)
            if isinstance(data, list):
                decisions = data
            elif isinstance(data, dict):
                # If the model wrapped the list in a key like "decisions" or "results"
                decisions = next(iter(data.values())) if data.values() else []
            else:
                decisions = []
                
            # 4. Filter and count
            supported_count = sum(1 for d in decisions if str(d).upper() == "YES")
            
        except Exception as e:
            # Fallback: keep your regex/string count for safety
            supported_count = response.upper().count('"YES"')

        # Metrics
        k = len(reference_claims)
        k_hat = len(hypothesis_claims)
        if k == 0: return 0.0, 0
        
        precision = min(1, float(supported_count / (supported_count + (k_hat - supported_count))) if (supported_count + (k_hat - supported_count)) > 0 else 0)
        recall = min(1, float(supported_count / k))
        
        if (precision + recall) == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        
        return score, supported_count

# --- 4. CONFIGURATION ---
MAX_WORKERS = 4  # Keep low for API stability
abstention_lock = threading.Lock() 

# --- 5. BATCH ABSTENTION (GPU) ---
def batch_detect_abstention(df, detector):
    print("   Running Abstention Detection (GPU Batch Mode)...")
    responses = df['generated_response'].fillna("").astype(str).tolist()
    inputs = [r[:4096] for r in responses]
    
    # Run in batches of 16
    raw_results = detector.classifier(inputs, batch_size=16, truncation=True)
    
    is_abstained_list = []
    for res in raw_results:
        label_id = int(res['label'].split('_')[-1])
        score = res['score']
        
        if label_id in [3, 5]:
            judgement = "PASS"
        elif label_id not in [3, 5] and score < 0.925:
            judgement = "PASS"
        else:
            judgement = "ABSTENTION"
        is_abstained_list.append(judgement == "ABSTENTION")
    return is_abstained_list

# --- 6. WORKERS ---
def worker_verify(index, row, afg, verifier, is_abstained, gt_facts):
    try:
        if is_abstained:
            return index, {
                "prompt": row['prompt'], "generated_response": row.get('generated_response', ''),
                "is_abstained": True, "score": 0.0, "supported_claims": 0,
                "afg_k_gen": 0, "afg_k_gt": len(gt_facts),
                "gen_facts": "[]", "gt_facts": json.dumps(gt_facts)
            }

        gen_response = row.get('generated_response', '')
        gen_facts, k_gen = afg.run(gen_response)
        
        # USE NEW BATCH VERIFIER
        accuracy_score, supported_count = verifier.verify(gen_facts, gt_facts)
        
        return index, {
            "prompt": row['prompt'], "generated_response": gen_response,
            "is_abstained": False, "score": accuracy_score, "supported_claims": supported_count,
            "afg_k_gen": k_gen, "afg_k_gt": len(gt_facts),
            "gen_facts": json.dumps(gen_facts), "gt_facts": json.dumps(gt_facts)
        }
    except Exception as e:
        return index, {"prompt": row.get('prompt', ''), "error": str(e), "score": 0.0}

# --- 7. CONTROLLER ---
def process_evaluation(df, input_filename, args, resources):
    print(f"\n📂 Processing: {input_filename} ({len(df)} rows)")
    afg, verifier, abstention_detector = resources
    
    # Merge GT
    if 'response_facts' not in df.columns and args.reference and os.path.exists(args.reference):
        print(f"   Merging facts from {args.reference}...")
        ref_df = pd.read_csv(args.reference)
        df['prompt'] = df['prompt'].astype(str)
        ref_df['prompt'] = ref_df['prompt'].astype(str)
        df = df.merge(ref_df[['prompt', 'response_facts']], on='prompt', how='left')
    
    # Phase 1: GPU Abstention
    is_abstained_mask = batch_detect_abstention(df, abstention_detector)
    
    # Phase 2: API Verification (Parallel)
    results_list = [None] * len(df)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {}
        for i, row in df.iterrows():
            gt_facts_raw = row.get('response_facts', None)
            gt_facts = []
            if gt_facts_raw:
                if isinstance(gt_facts_raw, str):
                    try: gt_facts = json.loads(gt_facts_raw)
                    except: gt_facts = [f.strip() for f in gt_facts_raw.split('\n') if f.strip()]
                else: gt_facts = gt_facts_raw

            future = executor.submit(worker_verify, i, row, afg, verifier, is_abstained_mask[i], gt_facts)
            future_to_idx[future] = i

        for future in tqdm(as_completed(future_to_idx), total=len(df), desc="   Verifying (Batch Mode)", unit="rows"):
            idx, res = future.result()
            results_list[idx] = res

    # Save
    final_results = [r for r in results_list if r is not None]
    scores = [r['score'] for r in final_results if 'score' in r]
    avg_score = sum(scores)/len(scores) if scores else 0
    print(f"   🏆 Average Score: {avg_score:.2%}")
    
    writer = ResultWriter()
    os.makedirs("eval_outputs", exist_ok=True)
    out_name = os.path.join("eval_outputs", f"eval_results_{os.path.splitext(input_filename)[0]}")  
    writer.write(final_results, base_filename=out_name)
    print(f"   ✅ Saved to {out_name}.csv")

def run_evaluate_batch(args):
    print(f"🚀 Turbo Batch Evaluate (v8) - Model: {args.model}")
    print("   (Loading GPU models once... please wait)")
    
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    abstention_detector = AbstentionDetector(device=args.device) 
    afg = AtomicFactGenerator(llm)
    
    # INJECT BATCH VERIFIER
    verifier = BatchFactVerifier(llm) 
    
    resources = (afg, verifier, abstention_detector)
    
    files = []
    if args.input_dir:
        files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    elif args.input:
        files = [args.input]
        
    for f in files:
        try:
            loader = DataLoader(f)
            df = loader.load()
            process_evaluation(df, os.path.basename(f), args, resources)
        except Exception as e:
            print(f"❌ Error {f}: {e}")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model", default="groq/llama-3.1-8b-instant")
    parent.add_argument("--device", default="cuda")
    parent.add_argument("--api_key", default=None)
    parent.add_argument("--mock", action="store_true")

    p_eval = subparsers.add_parser("evaluate", parents=[parent])
    p_eval.add_argument("--input", default=None)
    p_eval.add_argument("--input_dir", default=None)
    p_eval.add_argument("--reference", default=None)
    
    args = parser.parse_args()
    if args.command == "evaluate": run_evaluate_batch(args)