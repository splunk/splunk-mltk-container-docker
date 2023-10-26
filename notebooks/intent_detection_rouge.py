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
from sumeval.metrics.rouge import RougeCalculator
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











    
# In[6]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[8]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    return {}







    
# In[10]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    
    rouge = RougeCalculator(stopwords=True, lang="ja")
    inputs = df["actions"].values.tolist()
    intents = param['options']['params']['intents'].strip('"').split(',')
    ret = []
    for i in inputs:
        temp = []
        for intent in intents:
            rouge_1 = rouge.rouge_n(
                summary=i,
                references=intent,
                n=1)
            temp.append(rouge_1)
        
        result = intents[temp.index(max(temp))]

        # Scores and final outputs 
        ret.append(result)
    cols={'intent': ret}
    returns=pd.DataFrame(data=cols)
    return returns







    
# In[12]:


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







