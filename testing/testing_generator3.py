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

csv_path = os.path.join(os.path.dirname(__file__), '..', 'public_set_1.csv')
API_KEY = os.environ['GROQ_API_KEY']
MODEL_UNDER_TEST = 'groq/llama-3.1-8b-instant'

model = LLMWrapper(model_name=MODEL_UNDER_TEST, api_key=API_KEY)

genLoader = Generator_DataLoader(csv_path)
generator = Generator_Model(model=model, context='zero')

results = generator.run(genLoader)

try:
    genLoader.save(results, label=f'{MODEL_UNDER_TEST}_0SHOT')
except:
    print('placeholder save')
    genLoader.save(results, label='UNKNOWN_0SHOT')
