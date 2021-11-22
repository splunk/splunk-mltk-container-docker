#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
import pickle
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"









    
# In[5]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[12]:


# Create the random cut forest from the source data
def init(df,param):
    # Set model parameters
    X = df[param['feature_variables'][0]]    
    n_trees=10
    height=8
    window_size=250
    if 'options' in param:
        if 'params' in param['options']:
            if 'n_trees' in param['options']['params']:
                n_trees = int(param['options']['params']['n_trees'])
            if 'height' in param['options']['params']:
                height = int(param['options']['params']['height'])
            if 'window_size' in param['options']['params']:
                window_size = int(param['options']['params']['window_size'])
    
    # Create the half space tree
    model = compose.Pipeline(
        preprocessing.MinMaxScaler(),
        anomaly.HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=42)
    )

    return model









    
# In[15]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    X = df[param['feature_variables'][0]]
    # init with a few warm up samples
    for x in X[:10]:
        model = model.learn_one({'x': x})
    return len(X)







    
# In[22]:


# apply your model
# returns the calculated results
def apply(model,df,param):

    X = df[param['feature_variables'][0]]
    Y = []
    
    for x in X:
        features = {'x': x}
        model = model.learn_one(features)
        score = model.score_one(features)
        Y.append(score)        
        #print(f'Anomaly score for x={x:.3f}: {model.score_one(features):.3f}')

    # save the model
    if 'options' in param:
        if 'model_name' in param['options']:
            if 'params' in param['options']:
                if 'algo' in param['options']['params']:
                    name = param['options']['params']['algo'] + '_' + param['options']['model_name']
                    save(model,name)
                    #print('/apply updated and saved model with parameters ', model)
                    
    
    result=pd.DataFrame({'anomaly_score':Y})
    return result







    
# In[19]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + '.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model





    
# In[20]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + '.pkl', 'rb') as f:
        model = pickle.load(f)
    return model





    
# In[21]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns





