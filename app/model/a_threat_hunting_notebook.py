#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os

# for operationalization of the model we want to use a few other libraries later
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

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





















    
# In[12]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['encoder'] = OneHotEncoder(handle_unknown='ignore')
    model['detector'] = IsolationForest(contamination=0.01)
    return model































    
# In[26]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    features_to_encode = df[['ComputerName','EventCode']]
    model['encoder'].fit(features_to_encode)
    encoded_features = model['encoder'].transform(features_to_encode)
    df_encoded_features = pd.concat([df[['count']], pd.DataFrame(encoded_features.toarray()).add_prefix('f_')], axis=1)
    model['detector'].fit(df_encoded_features)
    info = {"message": "model trained"}
    return info







    
# In[28]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    features_to_encode = df[['ComputerName','EventCode']]
    encoded_features = model['encoder'].transform(features_to_encode)
    df_encoded_features = pd.concat([df[['count']], pd.DataFrame(encoded_features.toarray()).add_prefix('f_')], axis=1)
    outliers = model['detector'].predict(df_encoded_features)
    result = pd.DataFrame(outliers, columns=['outlier'])
    return result







    
# In[30]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    # we skip saving and loading in this example, but of course you can build your preferred serialization here
    #with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
    #    json.dump(model, file)
    return model





    
# In[31]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    # we skip saving and loading in this example, but of course you can build your preferred deserialization here
    model = {}
    #with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
    #    model = json.load(file)
    return model





    
# In[32]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







