# testing file
# to test components iteratively

# current test: AFG (HF small model only, so not API)
# testing with instruct model Gemma 3 270m on CPU
# requires auth with huggingface
# todo: test for a GPU model

# 21/1/26

import pandas as pd
# from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator

TEST_SAMPLE_SIZE = 5
sample_responses = list(pd.read_csv("public1_subset.csv")['response'][:TEST_SAMPLE_SIZE].values)

model = LLMWrapper(
    model_name = "google/gemma-3-270m-it",
    device = "cpu",
    api_key = None,
    temperature=0,
    mock = False
)

afg = AtomicFactGenerator(
    model=model
)

for response in sample_responses:
    atomised = afg.run(response)
    print(atomised)

# results with a small model seem ok at best. need to test with a larger model
# is temperature = 0 good? almost definitely
# temperature > 0.5 seems to actually bring hallucinations into results
# with artifacts from demons present in output
# which is interesting