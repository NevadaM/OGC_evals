import pandas as pd
import os
import json
from typing import List, Dict, Optional, Any, Iterator
from .logger import get_module_logger

logger = get_module_logger("data_loader")

class DataLoader:
    """
    Handles loading, validation, and batching of datasets for OGC Evals.
    
    Why use a DataLoader?
    1. **Schema Validation**: Ensures 'prompt', 'response', and 'generated_response' exist before processing starts.
    2. **Efficiency**: Instead of iterating row-by-row, `get_batches()` yields chunks of data. This allows 
       downstream components (like the AbstentionDetector pipeline) to process multiple inputs in parallel 
       (vectorisation), significantly reducing total runtime on GPUs.
    3. **Serialization**: Automatically deserializes pre-computed columns like 'response_facts'.
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
            
        logger.info(f"Loading dataset from {self.file_path}...")
        
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
        Ensures required columns exist, prepares 'generated_response', and deserializes facts.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Available: {list(self.df.columns)}")
        
        # Handle generated_response
        if 'generated_response' not in self.df.columns:
            logger.info("Notice: 'generated_response' column not found. Copying 'response' (ground truth) to 'generated_response' for pipeline testing.")
            self.df['generated_response'] = self.df['response'].copy()
        else:
            logger.info("Found 'generated_response' column. Using provided model outputs.")
            
        # Handle pre-computed facts (JSON deserialization)
        if 'response_facts' in self.df.columns:
            logger.info("Found 'response_facts' column. Deserializing JSON...")
            try:
                # We interpret the column as JSON strings. 
                # If it was saved by pandas list support, it might look like "['a', 'b']" (string representation)
                # which isn't valid JSON (single quotes). 
                # Ideally we use json.loads. If that fails, we might need ast.literal_eval (but json is safer).
                # We'll assume the 'prepare' step uses json.dumps().
                
                def safe_load(x):
                    if isinstance(x, list): return x # Already a list (e.g. from JSON/Parquet)
                    try:
                        return json.loads(x)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback for simple string representation if needed, or return empty
                        return []
                        
                self.df['response_facts'] = self.df['response_facts'].apply(safe_load)
            except Exception as e:
                logger.warning(f"Error deserializing 'response_facts': {e}")
        
        # Ensure no NaN values in critical columns
        cols_to_check = self.REQUIRED_COLUMNS + ['generated_response']
        if self.df[cols_to_check].isnull().any().any():
            logger.warning("Warning: Null values found in critical columns. Dropping incomplete rows.")
            self.df.dropna(subset=cols_to_check, inplace=True)
            
        logger.info(f"Dataset loaded successfully with {len(self.df)} rows.")

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
