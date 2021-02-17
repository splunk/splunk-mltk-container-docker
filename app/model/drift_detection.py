#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import datetime as dt
import bocd
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[4]:


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
    # model.fit()
    info = {"message": "model created"}
    return info







    
# In[62]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    p_hazard = 200
    p_mu=0
    p_kappa=1
    p_alpha=1
    p_beta=1
    if 'options' in param:
        if 'params' in param['options']:
            if 'hazard' in param['options']['params']:
                p_hazard = int(param['options']['params']['hazard'])
            if 'mu' in param['options']['params']:
                p_mu = float(param['options']['params']['mu'])
            if 'kappa' in param['options']['params']:
                p_kappa = float(param['options']['params']['kappa'])
            if 'alpha' in param['options']['params']:
                p_alpha = float(param['options']['params']['alpha'])
            if 'beta' in param['options']['params']:
                p_beta = float(param['options']['params']['beta'])
    target = param['target_variables'][0]
    signal = df[target].to_numpy()
    
    # Initialize object
    detector = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(p_hazard), bocd.StudentT(mu=p_mu, kappa=p_kappa, alpha=p_alpha, beta=p_beta))

    # Online estimation and get the maximum likelihood r_t at each time point
    rt_mle = np.empty(signal.shape)
    for i, d in enumerate(signal):
        detector.update(d)
        rt_mle[i] = detector.rt
    
    df['drift'] = 0
    index_changes = np.where(np.diff(rt_mle)<0)[0]
    for i in index_changes:
        df.at[i, 'drift'] = 1
   
    return df







    
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





