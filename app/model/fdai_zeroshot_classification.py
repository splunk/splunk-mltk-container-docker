#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple
from dataclasses import dataclass
# ...
def load_or_download_huggingface_model(model_name: str, model_dir: str):
    # Sanitize the repo ID to create a valid folder name
    repo_folder_name = model_name.replace("/", "--")
    local_repo_path = os.path.join(model_dir, repo_folder_name)
    config_file_path = os.path.join(local_repo_path, "config.json")

    model = None
    tokenizer = None

    if os.path.isdir(local_repo_path) and os.path.exists(config_file_path):
        print(f"Repo folder '{local_repo_path}' found locally and contains 'config.json'. Loading from local path.")
        ## NOTE: If the model is downloaded, proceed without validation
        # try:
        #     model = AutoModelForCausalLM.from_pretrained(local_repo_path)
        #     tokenizer = AutoTokenizer.from_pretrained(local_repo_path)
        #     print(f"Model and tokenizer loaded successfully from {local_repo_path}.")
        # except Exception as e:
        #     print(f"Error loading model from local path '{local_repo_path}': {e}")
        #     print("Attempting to re-download the model.")
        #     # Fallback to download if local load fails
        #     if os.path.exists(local_repo_path):
        #         import shutil
        #         shutil.rmtree(local_repo_path) # Clean up potentially corrupted local files
        #     model, tokenizer = _download_and_save_model(model_name, local_repo_path)
        model, tokenizer = {}, {}
    else:
        print(f"Repo folder '{local_repo_path}' not found or incomplete locally. Downloading from Hugging Face.")
        # 2. If not, load the model from huggingface and save in the local path
        model, tokenizer = _download_and_save_model(model_name, local_repo_path)

    return model, tokenizer

def _download_and_save_model(model_name: str, model_dir: str):
    """Helper function to download and save the model."""
    print(f"Downloading model '{model_name}' from Hugging Face...")
    try:
        # Create the local directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model and tokenizer downloaded and saved successfully to {model_dir}.")
        del model
        del tokenizer
        return {}, {}
    except Exception as e:
        print(f"Error downloading or saving model '{model_name}': {e}")
        return None, None

def load_fewshot_examples():
    """
    Helper function to read few-shot examples
    """
    file_path = "/srv/notebooks/data/fdai_zeroshot_classification.csv"
    if not os.path.isfile(file_path):
        return None

    df = pd.read_csv(file_path)
    
    required_columns = {'text', 'label'}
    if not required_columns.issubset(df.columns):
        return None
        
    tuple_list = list(df[['text', 'label']].itertuples(index=False, name=None))
    return tuple_list
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
MODEL_NAME = "fdtn-ai/Foundation-Sec-8B"

@dataclass
class ClassificationResult:
    """
    Stores the results of text classification

    Attributes:
        predicted_label: The most likely category (e.g., "malware", "phishing")
        confidence_score: How confident the model is (0-1, higher = more confident)
        perplexity_scores: Technical scores for each category
        raw_probabilities: Probability distribution across all categories
    """
    predicted_label: str
    confidence_score: float
    perplexity_scores: Dict[str, float]
    raw_probabilities: Dict[str, float]

class PerplexityClassifier:
    def __init__(self, model_name: str = "fdtn-ai/Foundation-Sec-8B",
                 labels: List[str] = None, device: str = "auto",
                 run_quantized: bool = False):
        self.labels = labels or []
        self.device = self._get_device(device)
        self.run_quantized = run_quantized
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if run_quantized:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            if self.device.type != 'cuda':
                self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_device(self, device: str) -> torch.device:
        """Automatically detect the best device (GPU vs CPU)"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def set_labels(self, labels: List[str]) -> None:
        """Set or update the classification categories"""
        self.labels = labels

    def _create_classification_prompt(self, text: str,
                                    few_shot_examples: List[Tuple[str, str]] = None) -> str:
        prompt_parts = [
            "This is a cybersecurity text classification task.",
            f"Available labels: {', '.join(self.labels)}",
            "Choose the most appropriate label for the given text.\n"
        ]

        # Add examples if provided
        if few_shot_examples:
            prompt_parts.append("Examples:")
            for example_text, example_label in few_shot_examples:
                prompt_parts.append(f'Text: """{example_text}"""')
                prompt_parts.append(f"Chosen label: {example_label}\n")

        # Add the text to classify
        prompt_parts.extend([
            f'Text: """{text}"""',
            "Chosen label:"
        ])

        return "\n".join(prompt_parts)

    def _calculate_batch_perplexities(self, prompt: str,
                                    completions: List[str]) -> Dict[str, float]:
        perplexities = {}

        for completion in completions:
            # Create full text: prompt + potential answer
            full_text = prompt + " " + completion

            # Convert text to model input
            inputs = self.tokenizer(full_text, return_tensors="pt",
                                  truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get length of just the prompt (not the completion)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt",
                                         truncation=True, max_length=2048)
            prompt_length = prompt_inputs["input_ids"].shape[1]

            # Calculate perplexity
            with torch.no_grad():  # Don't update model weights
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Focus only on the completion part
                completion_logits = logits[0, prompt_length-1:-1]
                completion_targets = inputs["input_ids"][0, prompt_length:]

                # Calculate cross-entropy loss (related to perplexity)
                loss = F.cross_entropy(completion_logits, completion_targets, reduction='mean')

                # Convert loss to perplexity
                perplexities[completion] = torch.exp(loss).item()

        return perplexities

    def classify(self, text: str,
                few_shot_examples: List[Tuple[str, str]] = None,
                return_all_scores: bool = False) -> ClassificationResult:
        if not self.labels:
            raise ValueError("No labels set. Use set_labels() to define classification labels.")

        # Create the classification prompt
        prompt = self._create_classification_prompt(text, few_shot_examples)

        # Calculate perplexity for each possible label
        perplexity_scores = self._calculate_batch_perplexities(prompt, self.labels)

        # Convert perplexities to probabilities (lower perplexity = higher probability)
        raw_probabilities = {
            label: 1.0 / perp if perp != float('inf') else 0.0
            for label, perp in perplexity_scores.items()
        }

        # Find the best (lowest perplexity) label
        best_label = min(perplexity_scores.keys(), key=lambda x: perplexity_scores[x])

        # Normalize probabilities to sum to 1
        total_inverse_perplexity = sum(raw_probabilities.values())
        normalized_probabilities = {
            label: prob / total_inverse_perplexity if total_inverse_perplexity > 0 else 1.0/len(self.labels)
            for label, prob in raw_probabilities.items()
        }

        # Calculate confidence as concentration of probability mass
        confidence_score = sum(p**2 for p in normalized_probabilities.values())

        return ClassificationResult(
            predicted_label=best_label,
            confidence_score=confidence_score,
            perplexity_scores=perplexity_scores if return_all_scores else {best_label: perplexity_scores[best_label]},
            raw_probabilities=normalized_probabilities if return_all_scores else {best_label: normalized_probabilities[best_label]}
        )

    def batch_classify(self, texts: List[str],
                      few_shot_examples: List[Tuple[str, str]] = None) -> List[ClassificationResult]:
        results = []
        for text in texts:
            result = self.classify(text, few_shot_examples)
            results.append(result)
        return results







    
# In[6]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[5]:


# This cell will run automatically to download the fdai model into the path /srv/app/model/data/fdtn-ai--Foundation-Sec-8B
# Model file will then be loaded directly from this path instead of redownloading
def init(df,param):
    model = {}
    _, _ = load_or_download_huggingface_model(MODEL_NAME, MODEL_DIRECTORY)
    return model







    
# In[5]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[7]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    try:
        labels = param['options']['params']['labels'].strip('"').split('&&')
    except:
        result = pd.DataFrame({'Message': ["ERROR: Please define input parameter \'labels\' as a list of labels separated by the && mark."]})
        return result
    try:
        X = df["text"].values.tolist()
    except:
        cols={'Message': ["ERROR: Please make sure you have a field in the search result named \'text\'"]}
        returns=pd.DataFrame(data=cols)
        return returns

    try:
        few_shot = int(param['options']['params']['few_shot'].strip('"'))
        if few_shot:
            few_shot_examples = load_fewshot_examples()
            print(few_shot_examples)
        else:
            few_shot_examples = None
    except:
        few_shot_examples = None
    
    classifier = PerplexityClassifier(model_name="/srv/app/model/data/fdtn-ai--Foundation-Sec-8B", labels=labels, run_quantized=True)
    p_label, p_prob, p_conf = [], [], []
    for i, text in enumerate(X, 1):
        result = classifier.classify(text, few_shot_examples, return_all_scores=True)
        p_label.append(result.predicted_label)
        p_prob.append(round(result.raw_probabilities[result.predicted_label],3))
        p_conf.append(round(result.confidence_score,3))

    cols = {"Label": p_label, "Probability": p_prob, "Confidence": p_conf}
    result = pd.DataFrame(data=cols)
    return result









    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[18]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







