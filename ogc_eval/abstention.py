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
