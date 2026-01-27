# testing file
# to test components iteratively

# current test: generator dataloader

# 27/1/26

# import pandas as pd
# from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.generator import Generator_DataLoader, Generator_Model
from ogc_eval.model import LLMWrapper

model = LLMWrapper(model_name="asdf", mock=True)

genLoader = Generator_DataLoader("randomised_sample_neil.csv")
generator = Generator_Model(model=model, context='zero')

results = generator.run(genLoader)

genLoader.save(results)