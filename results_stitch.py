# response generation doesn't actually stitch to a reference dataset
# so we need to make sure it does

import pandas as pd
import glob
import os

# 1. Load the Reference (Ground Truth + Facts)
reference_df = pd.read_csv("reference_dataset.csv")

# Ensure we have a common key. 
# If rows haven't been reordered, we can merge on index, but merging on 'prompt' is safer.
reference_df = reference_df.set_index("prompt")

# 2. Find your generation result files
result_files = glob.glob("*_generated_responses_*.csv")

print(f"Found {len(result_files)} result files to stitch.")

for file_path in result_files:
    print(f"Processing {file_path}...")
    
    # Load the generation result
    gen_df = pd.read_csv(file_path)
    
    # Check if we already have the facts (skip if so)
    if 'response_facts' in gen_df.columns:
        print(f"  - Already has facts. Skipping.")
        continue

    # 3. Merge
    # We map the 'response_facts' and 'response_facts_count' from Reference to Gen
    # using 'prompt' as the key.
    
    # Create a temporary merge dataframe to keep it clean
    merged_df = gen_df.merge(
        reference_df[['response_facts', 'response_facts_count']], 
        on='prompt', 
        how='left'
    )
    
    # 4. Save
    # We overwrite the file or save a new one ending in _ready.csv
    output_path = file_path.replace(".csv", "_ready.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"  - Saved stitched file to: {output_path}")

print("Done! You can now feed these *_ready.csv files into main.py")