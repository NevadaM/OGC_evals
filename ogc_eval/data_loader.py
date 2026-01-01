import pandas as pd
import os
from typing import List, Dict, Optional, Any, Iterator

class DataLoader:
    """
    Handles loading, validation, and batching of datasets for OGC Evals.
    
    Why use a DataLoader?
    1. **Schema Validation**: Ensures 'prompt', 'response', and 'generated_response' exist before processing starts.
    2. **Efficiency**: Instead of iterating row-by-row, `get_batches()` yields chunks of data. This allows 
       downstream components (like the AbstentionDetector pipeline) to process multiple inputs in parallel 
       (vectorisation), significantly reducing total runtime on GPUs.
    """
    REQUIRED_COLUMNS = ['prompt', 'response']
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Any = pd.DataFrame()
        
    def load(self) -> pd.DataFrame:
        """
        Loads the dataset, validates schema, and ensures 'generated_response' exists.
        Returns the raw pandas DataFrame.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        print(f"Loading dataset from {self.file_path}...")
        
        try:
            if self.file_path.lower().endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.lower().endswith('.jsonl'):
                self.df = pd.read_json(self.file_path, lines=True)
            elif self.file_path.lower().endswith('.json'):
                 self.df = pd.read_json(self.file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV, JSON, or JSONL.")
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")
            
        self._validate_and_prepare()
        return self.df
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Supports row-by-row iteration by default.
        Allows: `for row in loader: ...`
        """
        if self.df.empty:
            self.load()
            
        records = self.df.to_dict('records')
        for record in records:
            yield record

    def _validate_and_prepare(self):
        """
        Ensures required columns exist and prepares 'generated_response'.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Available: {list(self.df.columns)}")
        
        # Handle generated_response
        if 'generated_response' not in self.df.columns:
            print("Notice: 'generated_response' column not found. Copying 'response' (ground truth) to 'generated_response' for pipeline testing.")
            self.df['generated_response'] = self.df['response'].copy()
        else:
            print("Found 'generated_response' column. Using provided model outputs.")
            
        # Ensure no NaN values in critical columns
        if self.df[self.REQUIRED_COLUMNS + ['generated_response']].isnull().any().any():
            print("Warning: Null values found in critical columns. Dropping incomplete rows.")
            self.df.dropna(subset=self.REQUIRED_COLUMNS + ['generated_response'], inplace=True)
            
        print(f"Dataset loaded successfully with {len(self.df)} rows.")

    def get_batches(self, batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        """
        Yields the dataset in batches (list of dicts).
        
        Args:
            batch_size: Number of records per batch.
            
        Yields:
            List[Dict]: A list of row dictionaries (e.g., [{'prompt': '...', ...}, ...])
        """
        if self.df.empty:
            self.load()
            
        records = self.df.to_dict('records')
        for i in range(0, len(records), batch_size):
            yield records[i : i + batch_size]
