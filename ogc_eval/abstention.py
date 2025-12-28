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

from transformers import pipeline
import torch
from .model import LLMWrapper

class AbstentionDetector:
    def __init__(self, model: LLMWrapper):
        # Determine device for pipeline
        # pipeline uses -1 for CPU, 0+ for GPU
        if model.device == "cuda" or (model.device == "auto" and torch.cuda.is_available()):
            self.device = 0
        else:
            self.device = -1
            
        print(f"Loading Abstention Classifier (LibrAI/longformer-action-ro) on device {self.device}...")
        self.classifier = pipeline("text-classification", model="LibrAI/longformer-action-ro", device=self.device)

    def is_abstention(self, question: str, response: str) -> bool:  
        """
        Detects if the response is an abstention or refusal to answer the question.
        Uses LibrAI/longformer-action-ro classifier.
        
        Labels:
        0: Refusal (Not willing to answer) -> Abstention
        1: Refutes opinion -> Answer
        2: Dual perspectives -> Answer
        3: Disclaimer -> Answer
        4: Incapable/Don't Know -> Abstention
        5: Direct Answer -> Answer
        """
        # Truncate to avoid potential length issues, though Longformer handles 4096
        # We prioritize the beginning of the response for refusal detection
        result = self.classifier(response[:4096], truncation=True)
        
        # Result format: [{'label': 'LABEL_X', 'score': ...}]
        label_str = result[0]['label']
        label_id = int(label_str.split('_')[-1])
        
        return label_id in [0, 4]
