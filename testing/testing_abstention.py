# testing file
# to test components iteratively

# current test: abstention detection
# 21/01/26 -- with new datasetsample

import pandas as pd
# from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogc_eval.abstention import AbstentionDetector

TEST_SAMPLE_SIZE = 100
# sample_responses = list(pd.read_csv("datasetsample.csv")['response'][:TEST_SAMPLE_SIZE].values)

sample_responses = [
    "I apologize, but I cannot provide information regarding financial benefits or eligibility."
]

detector = AbstentionDetector()

for i in range(len(sample_responses)):
    answer = detector.is_abstention(sample_responses[i], verbose=True)
    print("Row ", i+2, " \t", answer)

# need to determine error rate / confidence intervals given custom rubric