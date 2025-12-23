## Atomised Fact Generation (AFG)
## To split either expected responses or 
## evaluated responses into lists of claims
## that can be enumerated and evaluated afterwards

## this is an almost exact reconstruction of SAFE's AFG
## although SAFE tries to make search queries
## whereas we need declarative statements a la FActScore
## in any case this works by using NLTK and spacy and an LLM
## to first split text into sentences
## then, upon retrieving examples from "demos" (or demons?)
## using them as few-shot examples to split sentences into 
## claims afterward

import json
import os
import re
import string
import nltk
from nltk import tokenize
import numpy as np
import rank_bm25
import spacy
from typing import List, Tuple, Dict, Any

from .model import LLMWrapper

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Load Spacy model
try:
    SPACY_MODEL = spacy.load('en_core_web_sm')
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please install it using `python -m spacy download en_core_web_sm`.")
    SPACY_MODEL = None

class AtomicFactGenerator:
    def __init__(self, model: LLMWrapper, demon_path: str = "ogc_eval/demos/demons.json"):
        self.nlp = SPACY_MODEL
        self.model = model
        self.demon_path = demon_path
        
        if os.path.exists(self.demon_path):
            with open(self.demon_path, 'r') as f:
                self.demons = json.load(f)
            
            tokenized_corpus = [doc.split(' ') for doc in self.demons.keys()]
            self.bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        else:
            print(f"Warning: Demon file not found at {self.demon_path}. AFG will run without few-shot examples.")
            self.demons = {}
            self.bm25 = None

    def run(self, text: str) -> Tuple[List[str], int]:
        """
        Splits the text into atomic facts.
        Returns a list of atomic facts and the count k.
        """
        if not text or not isinstance(text, str):
            return [], 0
            
        sentences = self._split_sentences(text)
        all_atoms = []
        
        for sentence in sentences:
            atoms = self._get_atoms_for_sentence(sentence)
            all_atoms.extend(atoms)
            
        return all_atoms, len(all_atoms)

    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting wrapper
        # Can be enhanced with the logic from the original atomic_facts.py (initials detection etc.)
        return tokenize.sent_tokenize(text)

    def _get_atoms_for_sentence(self, sentence: str) -> List[str]:
        prompt = self._construct_prompt(sentence)
        output = self.model.generate(prompt)
        return self._parse_output(output)

    def _construct_prompt(self, sentence: str) -> str:
        prompt = "Instructions:\n1. You are given a sentence. Your task is to break the sentence down into a list of atomic facts.\n2. An atomic fact is a sentence containing a singular piece of information.\n3. Each atomic fact in the outputted list should check a different piece of information.\n4. Use the previous examples to learn how to do this.\n5. You should only output the atomic facts as a list, with each item starting with \"- \". Do not include other formatting.\n6. Your task is to do this for the last sentence that is given.\n\n"
        
        if self.bm25:
            # Retrieve similar demons
            tokenized_query = sentence.split(' ')
            top_matches = self.bm25.get_top_n(tokenized_query, list(self.demons.keys()), n=3)
            
            for match in top_matches:
                prompt += f"Please breakdown the following sentence into independent facts: {match}\n"
                for fact in self.demons[match]:
                    prompt += f"- {fact}\n"
                prompt += "\n"
        
        prompt += f"Please breakdown the following sentence into independent facts: {sentence}\n"
        return prompt

    def _parse_output(self, output: str) -> List[str]:
        # Parse lines starting with "- "
        lines = output.strip().split('\n')
        facts = []
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:].strip())
            elif line.startswith("* "):
                 facts.append(line[2:].strip())
        return facts
