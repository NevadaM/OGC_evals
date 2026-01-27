## Automatic Fact Verification (AFV)
## checks a set of claims "a" made by a model
## against a set of claims "c" made in expected response
## and with LLM-as-a-judge sees how many in "a" are
## supported by "c". Reports the proportion of accurate
## claims at K -- K being the length of c.

## NEEDS CHECKING

import os
from typing import List, Tuple
from .model import LLMWrapper
from .logger import get_module_logger

logger = get_module_logger("afv")

class FactVerifier:
    def __init__(self, model: LLMWrapper):
        self.model = model
        self.prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        
        with open(os.path.join(self.prompt_dir, "afv_system.txt"), "r") as f:
            self.system_prompt = f.read().strip()
            
        with open(os.path.join(self.prompt_dir, "afv_user.txt"), "r") as f:
            self.user_prompt_template = f.read().strip()

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
            logger.debug(f"Verifying claim: {claim[:50]}...")
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template.format(reference_text=reference_text, claim=claim)}
            ]
            
            output = self.model.classify(messages, options=["YES", "NO"])
            if "YES" in output.upper():
                supported_count += 1
        
        # Precision@k calculation
        # k is the number of facts in the ground truth
        # we look to find the number of claims tested model makes that are backed up in ground truth (supported_count)
        # then cap that at k
        # more than k = too much info, repeating information, etc. that can bump up factuality measurements despite no positive utility change
        # 
        # 
        
        k = len(reference_claims)
        k_hat = len(hypothesis_claims)
        precision = float(supported_count / (supported_count + (k_hat - supported_count)))
        recall = min(1, float(supported_count / k))
        score =  2 * precision * recall / (precision + recall)
        
        return score, supported_count
