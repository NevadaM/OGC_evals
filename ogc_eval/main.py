import argparse
import sys
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from .logger import setup_logger, get_module_logger
from .data_loader import DataLoader
from .model import LLMWrapper
from .afg import AtomicFactGenerator
from .abstention import AbstentionDetector
from .afv import FactVerifier
from .result_writer import ResultWriter

# Initialize root logger with default settings
setup_logger()
logger = get_module_logger("main")

def run_prepare(args):
    """
    PREPARE MODE:
    Pre-computes atomic facts for the Ground Truth ('response') column.
    Saves a new CSV with a 'response_facts' column.
    """
    logger.info("Starting Dataset Preparation (AFG on Ground Truth)...")
    
    # Initialize Model (AFG requires LLM)
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    afg = AtomicFactGenerator(llm)
    
    # Load Data
    loader = DataLoader(args.input)
    df = loader.load()
    
    results = []
    
    # Process
    # We use iterrows or loader iteration. Since we need to modify the dataframe,
    # building a list and creating a new DF is cleaner.
    
    logger.info(f"Processing {len(df)} rows...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Atomizing GT"):
        response = row['response']
        facts, k = afg.run(response)
        
        # Create a new record preserving all original columns
        new_row = row.to_dict()
        # Serialize list to JSON string for CSV storage
        new_row['response_facts'] = json.dumps(facts)
        new_row['response_facts_count'] = k
        results.append(new_row)
        
    # Save
    result_df = pd.DataFrame(results)
    output_path = args.output or args.input.replace(".csv", "_prepped.csv")
    result_df.to_csv(output_path, index=False)
    logger.info(f"Preparation complete. Saved to {output_path}")


def run_evaluate(args):
    """
    EVALUATE MODE:
    Runs the full evaluation pipeline (Abstention -> AFG(Gen) -> AFV).
    Uses pre-computed 'response_facts' if available, otherwise computes them on the fly.
    """
    logger.info("Starting Evaluation Pipeline...")
    
    # Initialize Components
    # 1. LLM for AFG/AFV
    llm = LLMWrapper(model_name=args.model, device=args.device, api_key=args.api_key, mock=args.mock)
    
    # 2. Abstention Detector (PLM)
    abstention_detector = AbstentionDetector(device=args.device)
    
    # 3. AFG (for Generated Responses) & AFV
    afg = AtomicFactGenerator(llm)
    verifier = FactVerifier(llm)
    
    # 4. Data Loader & Result Writer
    loader = DataLoader(args.input)
    writer = ResultWriter()
    
    # Use batching for efficiency? 
    # Currently AFG and AFV are implemented for single items.
    # We can stick to row-by-row for clarity until components support batching.
    # However, DataLoader.get_batches() is available if we want to batch Abstention.
    
    # Let's do a hybrid approach: Batch Abstention (it supports it), then loop for AFG/AFV.
    batch_size = 16
    results = []
    
    logger.info(f"Evaluating in batches of {batch_size}...")
    
    for batch in tqdm(loader.get_batches(batch_size), desc="Evaluating"):
        # batch is a list of dicts
        
        # 1. Batch Abstention Detection
        # We need to extract responses.
        gen_responses = [row['generated_response'] for row in batch]
        
        # AbstentionDetector.classifier is a pipeline, it can take a list!
        # But our wrapper `is_abstention` takes a single string.
        # To leverage batching, we should technically update AbstentionDetector to accept lists.
        # For now, to avoid changing that file again, we loop.
        # (Optimization opportunity for later: Update AbstentionDetector to batch).
        
        for i, row in enumerate(batch):
            prompt = row['prompt']
            gen_response = row['generated_response']
            gt_facts = row.get('response_facts', [])
            
            # Fallback: if GT facts missing (not prepped), compute them
            if not gt_facts and 'response' in row:
                logger.debug("GT facts missing, computing on the fly...")
                gt_facts, _ = afg.run(row['response'])

            # 1. Abstention
            label_id, label_desc, score = abstention_detector.is_abstention(gen_response)
            is_abstained = label_id in [0, 4] # 0=Refusal, 4=Incapable
            
            row_result = {
                "prompt": prompt,
                "generated_response": gen_response,
                "is_abstained": is_abstained,
                "abstention_label": label_desc,
                "abstention_score": score,
                "score": 0.0, # Default
                "supported_claims": 0,
                "afg_k_gen": 0,
                "afg_k_gt": len(gt_facts)
            }
            
            if is_abstained:
                results.append(row_result)
                continue
                
            # 2. AFG (Generated)
            gen_facts, k_gen = afg.run(gen_response)
            
            # 3. AFV
            # Verify Generated Facts against Ground Truth Facts
            accuracy_score, supported_count = verifier.verify(gen_facts, gt_facts)
            
            row_result.update({
                "afg_k_gen": k_gen,
                "supported_claims": supported_count,
                "score": accuracy_score,
                "gen_facts": json.dumps(gen_facts), # Serialize for output CSV
                "gt_facts": json.dumps(gt_facts)    # Serialize for output CSV
            })
            
            results.append(row_result)

    # Write Results
    writer.write(results, base_filename="ogc_eval_results")


def main():
    parser = argparse.ArgumentParser(description="OGC Evals - Evaluation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parent_parser.add_argument("--model", type=str, default="gpt2", help="Model name (HF path or OpenAI model)")
    parent_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, auto)")
    parent_parser.add_argument("--api_key", type=str, default=None, help="OpenAI API Key")
    parent_parser.add_argument("--mock", action="store_true", help="Run in mock mode (no expensive calls)")

    # Prepare Command
    parser_prepare = subparsers.add_parser("prepare", parents=[parent_parser], help="Pre-compute atomic facts for Ground Truth")
    parser_prepare.add_argument("--output", type=str, default=None, help="Output path for prepared dataset")
    
    # Evaluate Command
    parser_eval = subparsers.add_parser("evaluate", parents=[parent_parser], help="Run evaluation pipeline")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        parser.print_help()
        # Default behavior if no command: print help and exit
        # Alternatively, could default to evaluate for backward compat, but explicit is better.

if __name__ == "__main__":
    main()