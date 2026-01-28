# script to feed prompts and personas from the dataset to a model for testing
# uses system prompts in prompts/

import json
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Iterator, Literal

from .model import LLMWrapper
from .data_loader import DataLoader
from .logger import get_module_logger

logger = get_module_logger("data_loader")

class Generator_DataLoader(DataLoader):
    """
    Handles loading, validation, and batching of datasets for testing a model.
    """

    REQUIRED_COLUMNS = ['prompt', 'targetAgeGroup', 'genderIdentity', 'educationBackground', 'targetProfession', 'digitalLiteracy', 'geoRegion', 'householdIncomeStatus']


    def __init__(self, file_path: str):
        super().__init__(file_path)

    def load(self) -> pd.DataFrame:
        return super().load()
    
    def save(self, results: List[str], label: str = 'LABEL', output_dir: str = '.') -> None:
        from datetime import datetime
        base_filename = f'{label}_generated_responses'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_base = f"{base_filename}_{timestamp}"
        csv_path = os.path.join(output_dir, f"{final_base}.csv")


        self.df['generated_response'] = results
        self.df.to_csv(csv_path, index=False)
        print('results saved to ', csv_path)

    
    def _validate_and_prepare(self):
        """
        Ensures required columns exist. Cleans non-required columns.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Available: {list(self.df.columns)}")

        self.df = self.df[self.REQUIRED_COLUMNS]
        if self.df.isnull().any().any():
            logger.warning("Warning: Null values found in critical columns. Dropping incomplete rows.")
            self.df.dropna(inplace=True)
            
        logger.info(f"Dataset loaded successfully with {len(self.df)} rows.")

    def get_batches(self, batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        return super().get_batches(batch_size)

class Generator_Model():
    def __init__(self, model: LLMWrapper, context: Literal["zero", "few"] = "zero"):
        self.model = model
        self.prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self.context = context
        self.examples = []

        with open(os.path.join(self.prompt_dir, "test_system.txt"), "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

        if self.context == 'few':
            with open(os.path.join(self.prompt_dir, "few_shot_test_examples.json"), "r", encoding="utf-8") as f:
                self.examples = json.load(f)

    def run(self, dataloader: Generator_DataLoader, batch_size=16):
        results = []
        dataloader.load()

        batches = dataloader.get_batches(batch_size)

        for batch in tqdm(batches):
            for i, row in enumerate(batch):
                row_result = self.single_row_run(row)

                results.append(row_result)

        return results
    
    def single_row_run(self, row) -> str:
        messages = self._construct_messages(row)
        output = self.model.generate(messages)
        # any parsing necessary? would go here
        return output

    def _construct_messages(self, row) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]

        if self.context == 'few':
            for example in self.examples:
                user_msg = (
                    'prompt: ' + str(example['prompt']) +'\n' + 
                    'targetAgeGroup: ' + str(example['targetAgeGroup']) +'\n' + 
                    'genderIdentity: ' + str(example['genderIdentity']) +'\n' + 
                    'educationBackground: ' + str(example['educationBackground']) +'\n' + 
                    'targetProfession: ' + str(example['targetProfession']) +'\n' + 
                    'digitalLiteracy: ' + str(example['digitalLiteracy']) +'\n' + 
                    'geoRegion: ' + str(example['geoRegion']) +'\n' + 
                    'householdIncomeStatus: ' + str(example['householdIncomeStatus']) +'\n' 
                )

                messages.append({"role": "user", "content": user_msg})

                assistant_msg = (
                    'response: ' + str(example['response']) + '\n'
                )
                
                messages.append({"role": "assistant", "content": assistant_msg})

        user_msg = (
            'prompt: ' + str(row['prompt']) +'\n' + 
            'targetAgeGroup: ' + str(row['targetAgeGroup']) +'\n' + 
            'genderIdentity: ' + str(row['genderIdentity']) +'\n' + 
            'educationBackground: ' + str(row['educationBackground']) +'\n' + 
            'targetProfession: ' + str(row['targetProfession']) +'\n' + 
            'digitalLiteracy: ' + str(row['digitalLiteracy']) +'\n' + 
            'geoRegion: ' + str(row['geoRegion']) +'\n' + 
            'householdIncomeStatus: ' + str(row['householdIncomeStatus']) +'\n' 
        )

        messages.append({"role": "user", "content": user_msg})

        return messages


