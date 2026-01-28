import argparse
import os
import json
import pandas as pd
import time
import random
import threading
import logging
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

# --- 2. MONKEY-PATCH LITELLM (SILENT RETRIES) ---
import litellm
from litellm import completion as original_completion

litellm.suppress_debug_info = True
litellm.drop_params = True
_logger = logging.getLogger("litellm")
_logger.setLevel(logging.CRITICAL)
_logger.propagate = False 

def robust_completion(*args, **kwargs):
    max_retries = 15
    attempt = 0
    while attempt < max_retries:
        try:
            return original_completion(*args, **kwargs)
        except litellm.RateLimitError:
            attempt += 1
            wait = 5 + (attempt * 2) + random.uniform(1, 3)
            if attempt == 5:
                tqdm.write(f"   (Thread stalling due to Rate Limits... auto-retrying)")
            time.sleep(wait)
        except Exception as e:
            raise e
    raise Exception("Max Retries Exceeded")

litellm.completion = robust_completion

# --- 3. CONFIGURATION ---
MAX_WORKERS = 4
abstention_lock = threading.Lock() 

# --- 4. PREPARE MODE ---
def worker_prepare(index, row, afg):
    try:
        response = row.get('response', row.get('ground_truth', ''))
        if not response or pd.isna(response): return index, [], 0
        facts, k = afg.run(response)
        return index, facts, k
    except: return index, [], 0

def run_prepare(args):
    print(f"🚀 Turbo Prepare: {args.model}")
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    afg = AtomicFactGenerator(llm)
    loader = DataLoader(args.input)
    df = loader.load()
    
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(worker_prepare, i, row, afg): i for i, row in df.iterrows()}
        for future in tqdm(as_completed(future_to_idx), total=len(df), desc="Atomizing"):
            idx, facts, k = future.result()
            row_data = df.iloc[idx].to_dict()
            row_data['response_facts'] = json.dumps(facts)
            row_data['response_facts_count'] = k
            results[idx] = row_data

    result_df = pd.DataFrame(results)
    output_path = args.output or args.input.replace(".csv", "_reference.csv")
    result_df.to_csv(output_path, index=False)
    print(f"✅ Saved Reference Dataset to {output_path}")

# --- 5. EVALUATE MODE ---
def worker_evaluate(index, row, afg, verifier, abstention_detector):
    try:
        gen_response = row.get('generated_response', '')
        
        # GT Facts Logic
        gt_facts_raw = row.get('response_facts', None)
        gt_facts = []
        if gt_facts_raw:
            if isinstance(gt_facts_raw, str):
                try: gt_facts = json.loads(gt_facts_raw)
                except: gt_facts = [f.strip() for f in gt_facts_raw.split('\n') if f.strip()]
            else: gt_facts = gt_facts_raw
        
        # Fallback if GT missing (Costly!)
        if not gt_facts and 'response' in row:
             gt_facts, _ = afg.run(row['response'])

        with abstention_lock:
            judgement = abstention_detector.is_abstention(gen_response, verbose=False)
        is_abstained = (judgement == "ABSTENTION")
        
        row_result = {
            "prompt": row['prompt'], "generated_response": gen_response,
            "is_abstained": is_abstained, "score": 0.0, "supported_claims": 0,
            "afg_k_gen": 0, "afg_k_gt": len(gt_facts),
            "gen_facts": "[]", "gt_facts": json.dumps(gt_facts)
        }

        if is_abstained: return index, row_result

        gen_facts, k_gen = afg.run(gen_response)
        accuracy_score, supported_count = verifier.verify(gen_facts, gt_facts)
        
        row_result.update({
            "afg_k_gen": k_gen, "supported_claims": supported_count,
            "score": accuracy_score, "gen_facts": json.dumps(gen_facts)
        })
        return index, row_result

    except Exception as e:
        return index, {"prompt": row.get('prompt', ''), "error": str(e), "score": 0.0}

def run_evaluate(args):
    print(f"🚀 Turbo Evaluate: {args.model}")
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    abstention_detector = AbstentionDetector(device=args.device)
    afg = AtomicFactGenerator(llm)
    verifier = FactVerifier(llm)
    
    loader = DataLoader(args.input)
    df = loader.load()
    
    # --- MERGING LOGIC ---
    if 'response_facts' not in df.columns:
        ref_path = args.reference
        
        # Auto-detect if not provided
        if not ref_path:
            possible_defaults = ["public_set_1_reference.csv", "reference_dataset.csv"]
            for p in possible_defaults:
                if os.path.exists(p):
                    ref_path = p
                    break
        
        if ref_path and os.path.exists(ref_path):
            print(f"   Merging with reference file: {ref_path}")
            ref_df = pd.read_csv(ref_path)
            # Merge on prompt
            df = df.merge(ref_df[['prompt', 'response_facts']], on='prompt', how='left')
        else:
            print("⚠️ WARNING: No reference facts found! Facts will be generated on-the-fly (Slower).")
            print("   Tip: Use --reference public_set_1_reference.csv")

    results_list = [None] * len(df)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(worker_evaluate, i, row, afg, verifier, abstention_detector): i 
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(future_to_idx), total=len(df), desc="Judging"):
            idx, res = future.result()
            results_list[idx] = res

    final_results = [r for r in results_list if r is not None]
    scores = [r['score'] for r in final_results if 'score' in r]
    if scores: print(f"\n🏆 Average Factuality Score: {sum(scores)/len(scores):.2%}")

    writer = ResultWriter()
    # Generate output filename based on input
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    out_name = f"eval_results_{input_base}"
    writer.write(final_results, base_filename=out_name)
    print(f"✅ Results written to {out_name}...")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--input", required=True)
    parent.add_argument("--model", default="groq/llama-3.3-70b-versatile")
    parent.add_argument("--device", default="cpu")
    parent.add_argument("--api_key", default=None)
    parent.add_argument("--mock", action="store_true")

    p_prep = subparsers.add_parser("prepare", parents=[parent])
    p_prep.add_argument("--output", default=None)
    
    p_eval = subparsers.add_parser("evaluate", parents=[parent])
    # NEW ARGUMENT
    p_eval.add_argument("--reference", default=None, help="Path to reference CSV containing ground truth facts")
    
    args = parser.parse_args()
    if args.command == "prepare": run_prepare(args)
    elif args.command == "evaluate": run_evaluate(args)