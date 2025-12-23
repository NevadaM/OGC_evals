## wrapper object for LLMs
## this is generic and can be used for models across the architecture:
## 1. for a model to generate the dataset
## 2. for a model to be evaluated against the dataset
## 3. for a model to perform AFG
## 4. for a model to perform AFV (with classify method)
## 5. for a model to do abstention detection (if necessary? with classify method)

## currently only mock architecture
## needs to take HuggingFace transformer OR an API key
## and, importantly, it needs to facilitate different ways of 
## having system prompts and user prompts, with templating

from typing import List, Optional
import os

class LLMWrapper:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # In a real scenario, we would load the model here.
        # import torch
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print(f"Initialized LLMWrapper with model: {model_name} on {device}")

    def generate(self, prompt: str, max_new_tokens: int = 256, stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generates text based on the prompt. 
        Since we don't have GPU access, this is a mock implementation.
        """
        # Logic for real generation:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Mock behavior for testing architecture
        print(f"--- [Mock LLM Call] ---\nPrompt start: {prompt[:100]}...\n-----------------------")
        return "- Mock atomic fact 1\n- Mock atomic fact 2"

    def classify(self, prompt: str, options: List[str]) -> str:
        """
        Classifies the input into one of the options.
        """
        # Mock behavior
        print(f"--- [Mock LLM Classify] ---\nPrompt start: {prompt[:100]}...\n-----------------------")
        
        # Simple heuristic for testing pipeline:
        # If checking for abstention ("Did the AI refuse..."), say NO (it didn't refuse, so we proceed).
        if "Did the AI refuse" in prompt:
            return "NO"
        
        # If checking for entailment ("Does the premise entail..."), say YES (it entails, so we get a score).
        if "Does the premise entail" in prompt:
            return "YES"

        return options[0]
