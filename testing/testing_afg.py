# testing file
# to test components iteratively

# current test: AFG (HF small model only, so not API)
# testing with instruct model Gemma 3 270m on CPU
# requires auth with huggingface
# todo: test for a GPU model

# 1/1/26

# import pandas as pd
# from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator

model = LLMWrapper(
    model_name = "google/gemma-3-270m-it",
    device = "cpu",
    api_key = None,
    mock = False
)

afg = AtomicFactGenerator(
    model=model
)

sample_responses = [
    "Yes, you can offer part-time jobs to people arriving from Ukraine.",
    "If both you and your partner were born on or after 6 April 1935, you may be able to claim Marriage Allowance instead of Married Couple’s Allowance.",
    "No, it is against the law to only check people you think are not British citizens. You must not discriminate against anyone based on their nationality or where they are from.",
    "Yes, you can use an expired Irish passport or passport card to prove your right to rent in England if you are an Irish citizen.",
    "You can register your interest in the Homes for Ukraine scheme by visiting the relevant link for your region (England, Northern Ireland, Scotland, or Wales). You must be over 18 and able to offer a spare room or home for at least 6 months.",
    # ""
]

for response in sample_responses:
    atomised = afg.run(response)
    print(atomised)

# NEED TO SORT OUT DEMOS vs DEMONS SYNTAX
# results with a small model seem ok at best. need to test with a larger model, but would also like
# to refine demos (or demons) to see what can be done.
# for example, if the response starts with "yes, you can", I think the atomised claim should just be
# "you can". which is ironed out with few-shot, hopefully.
# the code runs. I'd be interested in seeing more logging.