## Automatic Fact Verification (AFV)
## checks a set of claims "a" made by a model
## against a set of claims "c" made in expected response
## and with LLM-as-a-judge sees how many in "a" are
## supported by "c". Reports the proportion of accurate
## claims at K -- K being the length of c.

## NEEDS CHECKING

from typing import List, Tuple
from .model import LLMWrapper

class FactVerifier:
    def __init__(self, model: LLMWrapper):
        self.model = model

    def verify(self, hypothesis_claims: List[str], reference_claims: List[str]) -> Tuple[float, int]:
        """
        Verifies the hypothesis claims against the reference claims.
        Returns a score (Accuracy@k) and the number of supported claims.
        """
        if not hypothesis_claims:
            return 0.0, 0

        supported_count = 0
        
        # We check each hypothesis claim to see if it is entailed by ANY of the reference claims (or the set of them).
        # Context: "The number of claims in ai that are present in the ground truth ci"
        
        # We can pass all reference claims as context.
        reference_text = "\n".join([f"- {c}" for c in reference_claims])
        
        for claim in hypothesis_claims:
            messages = [
                {"role": "system", "content": "You are an intelligent fact-checking system. Determine if the hypothesis is entailed by the premise."},
                {"role": "user", "content": f"""Premise:
{reference_text}

Hypothesis:
{claim}

Does the premise entail the hypothesis? That is, is the hypothesis supported by the facts in the premise?
Answer only with "YES" or "NO"."""}
            ]
            
            output = self.model.classify(messages, options=["YES", "NO"])
            if "YES" in output.upper():
                supported_count += 1
        
        # Accuracy@k calculation
        # "where we don't count accurate claims in ai beyond the number of claims in ci"
        # This sounds like: min(supported_count, len(reference_claims)) / len(reference_claims) ?
        # Or maybe the user meant Precision? "number of claims in ai that are present in..."
        # If I generate 100 facts and 5 are true, is that good?
        # The prompt says: "accuracy@k, where we don't count accurate claims in ai beyond the number of claims in ci."
        # This suggests capping the numerator at k (where k = len(reference_claims)).
        
        k = len(reference_claims)
        if k == 0:
            return 0.0, 0 # If ground truth is empty, we can't support anything? Or maybe everything is hallucination.
            
        capped_supported = min(supported_count, k)
        score = capped_supported / k
        
        return score, supported_count
