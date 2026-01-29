import argparse
import os
import json
import pandas as pd
import time
import random
import threading
import glob
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

# --- 2. MONKEY-PATCH LITELLM (Silent & Robust) ---
import litellm
from litellm import completion as original_completion

litellm.suppress_debug_info = True
litellm.drop_params = True
_logger = logging.getLogger("litellm")
_logger.setLevel(logging.CRITICAL)
_logger.propagate = False 

def robust_completion(*args, **kwargs):
    """
    Catches RateLimits (429) AND Overload/Service Errors (503, 529).
    """
    max_retries = 15
    attempt = 0
    while attempt < max_retries:
        try:
            return original_completion(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for Rate Limit OR Server Overload (Anthropic/Google often send 503/Overloaded)
            if any(x in error_str for x in ["rate limit", "429", "503", "service unavailable", "overloaded"]):
                attempt += 1
                wait = 5 + (attempt * 2) + random.uniform(1, 3)
                if attempt == 5:
                    tqdm.write(f"   (Thread stalling: {error_str[:30]}... auto-retrying)")
                time.sleep(wait)
                continue
            
            # Real error? Raise it.
            raise e
    raise Exception("Max Retries Exceeded")

litellm.completion = robust_completion

# --- 3. CONFIGURATION ---
MAX_WORKERS = 20  # Keep low for API stability
abstention_lock = threading.Lock() 

# --- 4. SHARED WORKER FUNCTIONS ---

def worker_evaluate(index, row, afg, verifier, abstention_detector):
    try:
        gen_response = row.get('generated_response', '')
        
        # 1. Get GT Facts (Fastest: from row; Slowest: generate on fly)
        gt_facts_raw = row.get('response_facts', None)
        gt_facts = []
        if gt_facts_raw:
            if isinstance(gt_facts_raw, str):
                try: gt_facts = json.loads(gt_facts_raw)
                except: gt_facts = [f.strip() for f in gt_facts_raw.split('\n') if f.strip()]
            else: gt_facts = gt_facts_raw
        
        if not gt_facts and 'response' in row:
             gt_facts, _ = afg.run(row['response'])

        # 2. Abstention (Thread-Locked GPU call)
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

        # 3. Verify
        gen_facts, k_gen = afg.run(gen_response)
        accuracy_score, supported_count = verifier.verify(gen_facts, gt_facts)
        
        row_result.update({
            "afg_k_gen": k_gen, "supported_claims": supported_count,
            "score": accuracy_score, "gen_facts": json.dumps(gen_facts)
        })
        return index, row_result

    except Exception as e:
        return index, {"prompt": row.get('prompt', ''), "error": str(e), "score": 0.0}

def worker_prepare(index, row, afg):
    try:
        response = row.get('response', row.get('ground_truth', ''))
        if not response or pd.isna(response): return index, [], 0
        facts, k = afg.run(response)
        return index, facts, k
    except: return index, [], 0

# --- 5. LOGIC CONTROLLERS ---

def process_evaluation(df, input_filename, args, resources):
    """
    Handles the loop for a SINGLE file, using pre-loaded resources.
    """
    print(f"\n📂 Processing: {input_filename}")
    afg, verifier, abstention_detector = resources
    
    # Merge Reference Facts if missing
    if 'response_facts' not in df.columns and args.reference:
        if os.path.exists(args.reference):
            print(f"   Merging facts from {args.reference}...")
            ref_df = pd.read_csv(args.reference)
            # Ensure prompt columns match types (strings)
            df['prompt'] = df['prompt'].astype(str)
            ref_df['prompt'] = ref_df['prompt'].astype(str)
            
            df = df.merge(ref_df[['prompt', 'response_facts']], on='prompt', how='left')
        else:
            print(f"⚠️ Warning: Reference file {args.reference} not found.")

    results_list = [None] * len(df)
    
    # Run Threads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(worker_evaluate, i, row, afg, verifier, abstention_detector): i 
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(future_to_idx), total=len(df), desc=f"Evaluating {input_filename[:15]}...", unit="rows"):
            idx, res = future.result()
            results_list[idx] = res

    # Write Output
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
    print(f"🚀 Turbo Batch Evaluate initialized with model: {args.model}")
    print("   (Loading GPU models once... please wait)")
    
    # 1. Load Resources ONCE
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    abstention_detector = AbstentionDetector(device=args.device) # Heavy GPU Load
    afg = AtomicFactGenerator(llm)
    verifier = FactVerifier(llm)
    
    resources = (afg, verifier, abstention_detector)
    
    # 2. Determine File List
    files_to_process = []
    if args.input_dir:
        # Find all CSVs in directory
        pattern = os.path.join(args.input_dir, "*.csv")
        files_to_process = glob.glob(pattern)
        print(f"   Found {len(files_to_process)} files in {args.input_dir}")
    elif args.input:
        files_to_process = [args.input]
        
    if not files_to_process:
        print("❌ No input files found.")
        return

    # 3. Loop through files
    for file_path in files_to_process:
        try:
            loader = DataLoader(file_path)
            df = loader.load()
            filename = os.path.basename(file_path)
            
            process_evaluation(df, filename, args, resources)
            
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

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

# --- 6. CLI ---
if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model", default="groq/llama-3.1-8b-instant")
    parent.add_argument("--device", default="cuda") # Default to CUDA for SageMaker
    parent.add_argument("--api_key", default=None)
    parent.add_argument("--mock", action="store_true")

    # Prepare
    p_prep = subparsers.add_parser("prepare", parents=[parent])
    p_prep.add_argument("--input", required=True)
    p_prep.add_argument("--output", default=None)
    
    # Evaluate (Batch Compatible)
    p_eval = subparsers.add_parser("evaluate", parents=[parent])
    p_eval.add_argument("--input", default=None, help="Single CSV file")
    p_eval.add_argument("--input_dir", default=None, help="Directory containing CSVs to batch process")
    p_eval.add_argument("--reference", default=None, help="Reference CSV with ground truth facts")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "evaluate":
        run_evaluate_batch(args)