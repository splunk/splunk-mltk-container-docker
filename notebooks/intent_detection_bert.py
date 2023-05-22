#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"









    
# In[3]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param











    
# In[7]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[14]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    class IntentExtimator:
        def __init__(
          self, 
          intents,
          model_name = "/srv/app/model/data/classification/jp/bert_classification_jp",
          ):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.intent_dict = self.get_dict(intents)

        def tokenizing(self, text):
            tokens = self.tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            return input_ids, attention_mask

        def get_embedding(self, text):
            input_ids, attention_mask = self.tokenizing(text)
            with torch.no_grad():
              outputs = self.model(input_ids, attention_mask=attention_mask)
              embeddings = outputs.last_hidden_state[:, 0, :][0]
            return embeddings

        def get_dict(self, intents):
            '''
            create a dictionary of the embeddings for all intents
            '''
            ret = []
            for intent in intents:
              ret.append(self.get_embedding(intent).view(1, -1))
            return ret

        def get_scores(self, input):
            '''
            embed the input and calculate scores with all the intents in the dictionary
            '''
            scores = []
            input = self.get_embedding(input).view(1, -1)
            for i in self.intent_dict:
              score = torch.cosine_similarity(input, i)
              scores.append(score.item())
            return scores
    estimator = IntentExtimator(param['options']['params']['intents'].strip('"').split(','))
    l = estimator.intent_dict
    torch.save(l, '/srv/app/model/data/intent_dict.pt')
    return {}







    
# In[16]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    class IntentApply:
        def __init__(
          self, 
          intent_dict,
          model_name = "/srv/app/model/data/classification/jp/bert_classification_jp",
          ):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.intent_dict = intent_dict

        def tokenizing(self, text):
            tokens = self.tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            return input_ids, attention_mask

        def get_embedding(self, text):
            input_ids, attention_mask = self.tokenizing(text)
            with torch.no_grad():
              outputs = self.model(input_ids, attention_mask=attention_mask)
              embeddings = outputs.last_hidden_state[:, 0, :][0]
            return embeddings

        def get_scores(self, input):
            '''
            embed the input and calculate scores with all the intents in the dictionary
            '''
            scores = []
            input = self.get_embedding(input).view(1, -1)
            for i in self.intent_dict:
              score = torch.cosine_similarity(input, i)
              scores.append(score.item())
            return scores
    
    loaded_dict = torch.load('/srv/app/model/data/intent_dict.pt')
    estimator = IntentApply(loaded_dict)
    inputs = df["actions"].values.tolist()
    intents = param['options']['params']['intents'].strip('"').split(',')
    ret = []
    for i in inputs:
        results = estimator.get_scores(i)
        result = intents[results.index(max(results))]

        # Scores and final outputs 
        ret.append(result)
    cols={'pred_intent': ret}
    returns=pd.DataFrame(data=cols)
    return returns







    
# In[18]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    model = {}
    return model





    
# In[19]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    return model





    
# In[20]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







