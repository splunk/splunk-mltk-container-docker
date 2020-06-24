#!/usr/bin/env python
# coding: utf-8


    
# In[9]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import umap
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[11]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[13]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[18]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[20]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    X = df[param['feature_variables']]
    p = {
        "n_neighbors": 15,
        "n_components": 2
    }
    min_confidence = 0.0
    if 'options' in param:
        if 'params' in param['options']:
            for k in p.keys():
                if k in param['options']['params']:
                    p[k] = param['options']['params'][k]
    
    #reducer = umap.UMAP(random_state=42)
    reducer = umap.UMAP(
        random_state=42, 
        n_neighbors=int(p['n_neighbors']),
        n_components=int(p['n_components'])
    )

    embedding = reducer.fit_transform(X)
    result = pd.DataFrame(embedding)
    return result







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







