## wrapper object for LLMs
## this is generic and can be used for models across the architecture:
## 1. for a model to generate the dataset
## 2. for a model to be evaluated against the dataset
## 3. for a model to perform AFG
## 4. for a model to perform AFV (with classify method)
## 5. for a model to do abstention detection if necessary

## Supports:
## - Mock mode (default, for testing without GPU/API)
## - HuggingFace Transformers (local execution)
## - OpenAI API (via api_key or env var)
## 
## Handles chat templating for system/user prompts automatically where possible.

from typing import List, Optional, Union, Dict, Any
import os
from .logger import get_module_logger
from litellm import completion

logger = get_module_logger("model")

class LLMWrapper:
    def __init__(self, model_name: str, device: str = "cpu", api_key: Optional[str] = None, temperature: float = 0.1, mock: bool = False):
        self.model_name = model_name
        self.device = device
        self.api_key = api_key
        
        # Initialize attributes to None to satisfy type checkers
        self.model: Any = None
        self.tokenizer: Any = None
        self.client: Any = None
        self.temperature = temperature
        self.mode: str = "mock"

        if mock:
            self._set_mock_mode()
        elif self.api_key or any(x in model_name.lower() for x in ["gpt-3", "gpt-4", "claude", "gemini"]):
            self._init_api()
        else:
            self._init_hf()

    def _set_mock_mode(self):
        self.mode = "mock"
        logger.info(f"Initialized LLMWrapper in MOCK mode")

    def _init_api(self):
        self.mode = "api"
        logger.info(f"Initialized LLMWrapper in API mode (LiteLLM) with model: {self.model_name}")

    def _init_hf(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading HF model: {self.model_name} onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda" and self.device != "auto":
                self.model = self.model.to(self.device)
            
            self.mode = "hf"
            logger.info(f"Initialized LLMWrapper in HF mode with model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load HF model: {e}. Falling back to mock.")
            self._set_mock_mode()

    def generate(self, 
                 input_data: Union[str, List[Dict[str, str]]], 
                 max_new_tokens: int = 2000, 
                 stop_sequences: Optional[List[str]] = None,
                 return_input_data: bool = False) -> str:
        temperature = self.temperature
        if self.mode == "mock":
            return self._generate_mock(input_data)
        elif self.mode == "api":
            return self._generate_api(input_data, max_new_tokens, temperature, stop_sequences, return_input_data)
        elif self.mode == "hf":
            return self._generate_hf(input_data, max_new_tokens, temperature, return_input_data)
        return ""

    def _generate_mock(self, input_data: Any) -> str:
        prompt_preview = str(input_data)[:100]
        logger.debug(f"--- [Mock LLM Call] ---\nInput: {prompt_preview}...\n-----------------------")
        return "Output Output Output"

    def _generate_api(self, input_data: Union[str, List[Dict[str, str]]], max_tokens: int, temperature: float, stop: Optional[List[str]], return_input_data: bool) -> str:
        try:
            # LiteLLM expects standard OpenAI-format messages
            messages = [{"role": "user", "content": input_data}] if isinstance(input_data, str) else input_data
            
            # call litellm.completion instead of self.client.chat.completions.create
            response = completion(
                model=self.model_name, # Pass "gemini/gemini-pro" or "claude-3-opus" here directly
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                num_retries=3
            )
            
            # LiteLLM normalizes the output to match OpenAI's format exactly
            content = response.choices[0].message.content  # type: ignore

            ##### DEBUGGING EARLY STOPPING
            finish_reason = response.choices[0].finish_reason  # type: ignore
            if finish_reason != "stop":
                print(f"DEBUG: Generation cut off! Reason: {finish_reason} | Max Tokens: {max_tokens}")

            content = response.choices[0].message.content # type: ignore
            #####

            return content if content else ""
        
        except Exception as e:
            logger.error(f"LiteLLM Generation Error: {e}")
            return ""

    def _generate_hf(self, input_data: Union[str, List[Dict[str, str]]], max_new_tokens: int, temperature: float, return_input_data: bool) -> str:
        import torch
        
        if isinstance(input_data, list):
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(input_data, tokenize=False, add_generation_prompt=True)
            else:
                prompt = ""
                for msg in input_data:
                    prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                prompt += "ASSISTANT:"
        else:
            prompt = input_data

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        final = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if return_input_data:
            return "[INPUT " + input_data + "]" + final # type: ignore
        else:
            return final

    def classify(self, input_data: Union[str, List[Dict[str, str]]], options: List[str]) -> str:
        if self.mode == "mock":
            prompt_str = str(input_data)
            if "Did the AI refuse" in prompt_str: return "NO"
            if "Does the premise entail" in prompt_str: return "YES"
            return options[0]

        response = self.generate(input_data, max_new_tokens=10).strip()
        
        for opt in options:
            if response.lower() == opt.lower(): return opt
        for opt in options:
            if opt.lower() in response.lower(): return opt
                
        return response if response else options[0]
