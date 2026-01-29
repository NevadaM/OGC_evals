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

# --- 2. MONKEY-PATCH LITELLM (TRULY SILENT) ---
import litellm
from litellm import completion as original_completion

# Aggressive Silence
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("litellm").setLevel(logging.CRITICAL)

def robust_completion(*args, **kwargs):
    """
    Silent retry wrapper. Catches 429, 503, Timeout.
    Only raises exception if max_retries is exceeded.
    """
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 120.0 

    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            return original_completion(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            retry_triggers = ["rate limit", "429", "503", "service unavailable", "overloaded", "timeout", "timed out"]
            
            if any(x in error_str for x in retry_triggers):
                attempt += 1
                # Exponential Backoff (Silent)
                wait = 10 + (attempt * 5) + random.uniform(1, 5)
                time.sleep(wait)
                continue
            
            raise e
    raise Exception("Max Retries Exceeded")

litellm.completion = robust_completion

# --- 3. CONFIGURATION ---
MAX_WORKERS = 4  # Keep low for API stability

# --- 4. ABSTENTION BATCHER (GPU SPEEDUP) ---
def batch_detect_abstention(df, detector):
    """
    Runs the HuggingFace pipeline in batches on the GPU.
    Replicates the logic from ogc_eval/abstention.py
    """
    print("   Running Abstention Detection (Batch Mode)...")
    
    # Extract responses, handling NaN
    responses = df['generated_response'].fillna("").astype(str).tolist()
    # Truncate to 4096 chars (matches original logic)
    inputs = [r[:4096] for r in responses]
    
    # Run Pipeline (Fast!)
    # classifier returns list of dicts: [{'label': 'LABEL_X', 'score': Y.YY}, ...]
    raw_results = detector.classifier(inputs, batch_size=16, truncation=True)
    
    is_abstained_list = []
    
    for res in raw_results:
        label_str = res['label']
        score = res['score']
        label_id = int(label_str.split('_')[-1])
        
        # LOGIC REPLICATION from abstention.py
        if label_id in [3, 5]:
            judgement = "PASS"
        elif label_id not in [3, 5] and score < 0.925:
            judgement = "PASS"
        else:
            judgement = "ABSTENTION"
            
        is_abstained_list.append(judgement == "ABSTENTION")
        
    return is_abstained_list

# --- 5. WORKER (API ONLY) ---
def worker_verify(index, row, afg, verifier, is_abstained, gt_facts):
    """
    Only runs for NON-ABSTAINED rows.
    """
    try:
        gen_response = row.get('generated_response', '')
        
        row_result = {
            "prompt": row['prompt'], "generated_response": gen_response,
            "is_abstained": is_abstained, "score": 0.0, "supported_claims": 0,
            "afg_k_gen": 0, "afg_k_gt": len(gt_facts),
            "gen_facts": "[]", "gt_facts": json.dumps(gt_facts)
        }

        # If Abstained (shouldn't really happen if filtered, but safe to keep)
        if is_abstained: return index, row_result

        # 1. AFG (Generate Facts)
        gen_facts, k_gen = afg.run(gen_response)
        
        # 2. AFV (Verify Facts)
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

# --- 6. CONTROLLERS ---

def process_evaluation(df, input_filename, args, resources):
    print(f"\n📂 Processing: {input_filename} ({len(df)} rows)")
    afg, verifier, abstention_detector = resources
    
    # A. MERGE FACTS
    if 'response_facts' not in df.columns and args.reference:
        if os.path.exists(args.reference):
            print(f"   Merging facts from {args.reference}...")
            ref_df = pd.read_csv(args.reference)
            df['prompt'] = df['prompt'].astype(str)
            ref_df['prompt'] = ref_df['prompt'].astype(str)
            df = df.merge(ref_df[['prompt', 'response_facts']], on='prompt', how='left')
    
    # B. BATCH ABSTENTION (GPU)
    # This runs ONCE per file, super fast
    is_abstained_mask = batch_detect_abstention(df, abstention_detector)
    
    # C. PARALLEL VERIFICATION (API)
    results_list = [None] * len(df)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {}
        
        for i, row in df.iterrows():
            is_abstained = is_abstained_mask[i]
            
            # Prepare GT Facts for this row
            gt_facts_raw = row.get('response_facts', None)
            gt_facts = []
            if gt_facts_raw:
                if isinstance(gt_facts_raw, str):
                    try: gt_facts = json.loads(gt_facts_raw)
                    except: gt_facts = [f.strip() for f in gt_facts_raw.split('\n') if f.strip()]
                else: gt_facts = gt_facts_raw

            # If Abstained, we can skip the API call entirely and fill result now
            if is_abstained:
                results_list[i] = {
                    "prompt": row['prompt'], "generated_response": row.get('generated_response', ''),
                    "is_abstained": True, "score": 0.0, "supported_claims": 0,
                    "afg_k_gen": 0, "afg_k_gt": len(gt_facts),
                    "gen_facts": "[]", "gt_facts": json.dumps(gt_facts)
                }
            else:
                # Submit to API Worker
                future = executor.submit(worker_verify, i, row, afg, verifier, False, gt_facts)
                future_to_idx[future] = i

        # Process API results as they finish
        # Only non-abstained rows are in this loop, so the progress bar reflects "work to be done"
        if future_to_idx:
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="   Verifying", unit="rows"):
                idx, res = future.result()
                results_list[idx] = res
        else:
            print("   (All rows abstained - skipping verification)")

    # D. SAVE
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
    
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    abstention_detector = AbstentionDetector(device=args.device) 
    afg = AtomicFactGenerator(llm)
    verifier = FactVerifier(llm)
    
    resources = (afg, verifier, abstention_detector)
    
    files_to_process = []
    if args.input_dir:
        pattern = os.path.join(args.input_dir, "*.csv")
        files_to_process = glob.glob(pattern)
        print(f"   Found {len(files_to_process)} files in {args.input_dir}")
    elif args.input:
        files_to_process = [args.input]
        
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

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model", default="groq/llama-3.1-8b-instant")
    parent.add_argument("--device", default="cuda")
    parent.add_argument("--api_key", default=None)
    parent.add_argument("--mock", action="store_true")

    p_prep = subparsers.add_parser("prepare", parents=[parent])
    p_prep.add_argument("--input", required=True)
    p_prep.add_argument("--output", default=None)
    
    p_eval = subparsers.add_parser("evaluate", parents=[parent])
    p_eval.add_argument("--input", default=None)
    p_eval.add_argument("--input_dir", default=None)
    p_eval.add_argument("--reference", default=None)
    
    args = parser.parse_args()
    
    if args.command == "prepare": run_prepare(args)
    elif args.command == "evaluate": run_evaluate_batch(args)