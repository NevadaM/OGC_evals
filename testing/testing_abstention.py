# testing file
# to test components iteratively

# current test: abstention detection
# 28/12/25

# import pandas as pd
# from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.abstention import AbstentionDetector

sample_responses = [
    "Yes, you can offer part-time jobs to people arriving from Ukraine.",
    "If both you and your partner were born on or after 6 April 1935, you may be able to claim Marriage Allowance instead of Married Couple’s Allowance.",
    "No, it is against the law to only check people you think are not British citizens. You must not discriminate against anyone based on their nationality or where they are from.",
    "Yes, you can use an expired Irish passport or passport card to prove your right to rent in England if you are an Irish citizen.",
    "You can register your interest in the Homes for Ukraine scheme by visiting the relevant link for your region (England, Northern Ireland, Scotland, or Wales). You must be over 18 and able to offer a spare room or home for at least 6 months.",
    # ""
]

detector = AbstentionDetector()

for response in sample_responses:
    answer = detector.is_abstention(response)
    print(answer)

# returns 5s for everything except third answer, which is a 1 (refutes opinion)
# need to determine error rate / confidence intervals
# should ensure in expected response generation that models do not generate something
# that could be misconstrued as one of these.