#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import datetime as dt
import stumpy
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[36]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[38]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[40]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model created"}
    return info







    
# In[42]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    m = 24
    if 'options' in param:
        if 'params' in param['options']:
            if 'm' in param['options']['params']:
                m = int(param['options']['params']['m'])
    target = param['target_variables'][0]
    mp = stumpy.stump(df[target], m)    
    result = pd.DataFrame(mp[:, 0], columns=['matrix_profile'])
    return pd.concat([df, result], axis=1)







    
# In[33]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[34]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[35]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns





