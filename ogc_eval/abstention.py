## abstention detection
## currently goes before AFG
## as an LLM-as-a-judge which I don't think I want to use

## essentially needs to detect model responses to
## prompts that are dodging the question
## this is to avoid wasting time in afg/afv
## and also measure where LLMs are less confident

## what are our options for abstention detection?
## FActScore does it with simple keywords
## SAFE doesn't do it at all
## VeriScore doesn't really go beyonf FActScore
## "Do-not-answer" [https://arxiv.org/pdf/2308.13387] has a PLM-based classifier <600MB
## https://github.com/Libr-AI/do-not-answer/blob/main/notebooks/4_automatic_evaluation_PLM.ipynb 
## that classifies against five options

from typing import Union
from transformers import pipeline
import torch
from .logger import get_module_logger

logger = get_module_logger("abstention")

class AbstentionDetector:
    def __init__(self, device: str = "cpu"):

        self.ans_id_to_result = {
            0: "Refusal (Not willing to answer) -> Abstention",
            1: "Refutes opinion -> Abstention",
            2: "Dual perspectives -> Abstention",
            3: "Disclaimer -> Abstention",
            4: "Incapable/Don't Know -> Abstention",
            5: "Direct Answer -> Answer",
        }

        # Determine device for pipeline
        # pipeline uses -1 for CPU, 0+ for GPU
        if device == "cuda" and torch.cuda.is_available():
            self.device_id = 0
        elif device == "auto" and torch.cuda.is_available():
            self.device_id = 0
        else:
            self.device_id = -1
            
        logger.info(f"Loading Abstention Classifier (LibrAI/longformer-action-ro) on device {self.device_id}...")
        self.classifier = pipeline("text-classification", model="LibrAI/longformer-action-ro", device=self.device_id)

    def is_abstention(self, response: str, verbose=False) -> Union[str, tuple]:  
        """
        Detects if the response is an abstention or refusal to answer the question.
        Uses LibrAI/longformer-action-ro classifier.
        
        Labels:
        0: Refusal (Not willing to answer) -> Abstention
        1: Refutes opinion -> Abstention
        2: Dual perspectives -> Abstention
        3: Disclaimer -> Abstention
        4: Incapable/Don't Know -> Abstention
        5: Direct Answer -> Answer
        """
        # Truncate to avoid potential length issues, though Longformer handles 4096
        # We prioritize the beginning of the response for refusal detection
        result = self.classifier(response[:4096], truncation=True)
        
        # Result format: [{'label': 'LABEL_X', 'score': ...}]
        label_str = result[0]['label'] # type: ignore
        score = result[0]['score'] # type: ignore
        label_id = int(label_str.split('_')[-1])

        # Adjusted rubric
        # after tests of datasetsample, where everything is either 3, 5, or 1 (3 most common, 1 least)
        # disclaimers are fine, so everything with a 3 still passes
        # if not a 5 or 3, might still be ok, having manually reviewed all 1s. need 92.5% confidence that
        # the 1 is true. 92.5% threshold has been tested and seems to work well. 
        if label_id in [3, 5]:
            judgement = "PASS"
        elif label_id not in [3, 5] and score < 0.925:
            judgement = "PASS"
        else:
            judgement = "ABSTENTION"

        if verbose:
            return judgement, label_id, self.ans_id_to_result[label_id], score
        else:
            return judgement
