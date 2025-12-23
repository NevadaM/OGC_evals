import pandas as pd
import os
from typing import List, Dict, Optional, Any

class DataLoader:
    """
    Handles loading and validation of datasets for OGC Evals.
    Supports CSV and JSONL formats.
    """
    REQUIRED_COLUMNS = ['prompt', 'response']
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Any = pd.DataFrame
        
    def load(self) -> pd.DataFrame:
        """
        Loads the dataset and validates schema.
        Returns a pandas DataFrame.
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
            
        self._validate()
        return self.df
        
    def _validate(self):
        """
        Ensures required columns exist.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Available: {list(self.df.columns)}")
        
        if 'generated_response' not in self.df.columns:
            print("Notice: 'generated_response' column not found. Dataset expects live generation or mock generation.")
