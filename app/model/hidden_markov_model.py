#!/usr/bin/env python
# coding: utf-8


    
# In[186]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import pomegranate as pg
from pomegranate import *
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"









    
# In[189]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[191]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    #model['hmm'] = HiddenMarkovModel("HMM")
    return model







    
# In[205]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    X = df[param['feature_variables'][0]]

    distinct_states = set(''.join(X.unique()))
    equal_probability = 1.0 / len(distinct_states)
    discreet_equal_states = { state : equal_probability for state in distinct_states }
    discreet_equal_states_distribution = DiscreteDistribution( discreet_equal_states )
    equal_state = State( discreet_equal_states_distribution, name="equal_state" )

    #model = {}
    hmm_model = HiddenMarkovModel("HMM")
    hmm_model.add_states( [equal_state] )
    hmm_model.add_transition( hmm_model.start, equal_state, 1.00 )
    hmm_model.add_transition( equal_state, equal_state, 0.99 )
    hmm_model.add_transition( equal_state, hmm_model.end, 0.01)
    hmm_model.bake()

    info = hmm_model.fit( [ list(x) for x in X ] , max_iterations=10, n_jobs=6 )
    model['hmm'] = hmm_model
    model['info'] = info
    #info = {"message": "model trained"}

    return info







    
# In[199]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    field = param['feature_variables'][0]
    X = df[field]
    y_hat = X.apply(lambda x: model['hmm'].log_probability(list(x)))
    result = pd.DataFrame(y_hat).rename(columns={field: param['feature_variables'][0]+"_log_probability"})
    return result







    
# In[201]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        file.write(model['hmm'].to_json())
    return model







    
# In[203]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model_json = file.read()
        hmm = HiddenMarkovModel("HMM").from_json(model_json)
        model['hmm'] = hmm
    return model







    
# In[213]:


# return a model summary
def summary(model=None):    
    returns = {"version": {"pomegranate": pg.__version__ } }
    if model!=None:
        if 'info' in model:
            returns['info'] = model['info']
        elif 'hmm' in model:
            returns['info'] = model['hmm'].to_json()
    return returns









