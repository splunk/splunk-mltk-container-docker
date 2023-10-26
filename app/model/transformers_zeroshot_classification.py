#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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







    
# In[ ]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    return model







    
# In[ ]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[16]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    try:
        if param['options']['params']['lang'] == "jp":
            pipe = pipeline("zero-shot-classification", model="Formzu/bert-base-japanese-jsnli")
        else:
            pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except:
        pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    X = df.text.values
    candidate_labels = param['options']['params']['labels'].split("+")
    hypothesis_template = param['options']['params']['prompt']
    Y = []
    print(candidate_labels)
    print(hypothesis_template)
    for item in X:
        out = pipe(item, candidate_labels, hypothesis_template="This example is {}.")
        cand = out["labels"][0]
        cand_score = str(round(out["scores"][0], 2))
        Y.append(cand + " " + cand_score)

    cols={"label with score": Y}
    result=pd.DataFrame(data=cols)
    
    return result







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







