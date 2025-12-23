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

from .model import LLMWrapper

class AbstentionDetector:
    def __init__(self, model: LLMWrapper):
        self.model = model

    def is_abstention(self, question: str, response: str) -> bool:
        """
        Detects if the response is an abstention or refusal to answer the question.
        """
        prompt = f"""You are an evaluator checking if an AI assistant answered a user's question.
User Question: "{question}"
AI Response: "{response}"

Did the AI refuse to answer, or claim it cannot answer, or say it doesn't know?
Answer only with "YES" or "NO".
"""
        # In a real implementation, we'd use a more robust check or a classifier
        output = self.model.classify(prompt, options=["YES", "NO"])
        
        return "YES" in output.upper()
