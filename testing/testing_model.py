# testing file
# to test components iteratively

# current test: LLMwrapper object (HF small model only, so not API)
# testing with non-instruct model Gemma 3 270m on CPU
# requires auth with huggingface
# todo: test for a GPU model

# 30/12/25

# import pandas as pd
# from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.model import LLMWrapper

model = LLMWrapper(
    model_name = "google/gemma-3-270m",
    device = "cpu",
    api_key = None,
    mock = False
)

sample_input_data = "Once upon a time, "

output_data = model.generate(
    input_data = sample_input_data,
    return_input_data = True
)

print(output_data)

# works
# note that temperature of generation by default is 1
# note that only tested a cpu small model
# note that did not test API access