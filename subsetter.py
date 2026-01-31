import os
import pandas as pd

def process_csv_subsets(source_folder, dest_folder, row_limit=10):
    """
    Reads all CSV files in source_folder and saves the first 'row_limit' rows
    to dest_folder.
    """
    # 1. Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")

    # 2. Loop through all files in the source directory
    files_processed = 0
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".csv"):
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)

            try:
                # 3. Read only the first 'row_limit' rows
                # nrows limits the number of rows read (excluding header)
                df = pd.read_csv(source_path, nrows=row_limit)
                
                # 4. Save to the new folder
                # index=False ensures we don't add an extra numbered column
                df.to_csv(dest_path, index=False)
                
                print(f"Successfully processed: {filename} ({len(df)} rows)")
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nFinished! Processed {files_processed} files.")

# --- Configuration ---
# Update these paths to match your actual folders
# You can use absolute paths (e.g., 'C:/Users/Name/Data') or relative paths
SOURCE_DIRECTORY = 'fast_results'
DESTINATION_DIRECTORY = 'preliminary_results_10s'

if __name__ == "__main__":
    process_csv_subsets(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)