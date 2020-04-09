#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import datetime
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import dask_ml.cluster

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







    
# In[5]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    
    #client = Client("tcp://127.0.0.1:")
    client = Client(processes=False)
    
    model['dask_client'] = client
    return model







    
# In[18]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    ddf = dd.from_pandas(df, npartitions=4)  
    # features dataframe
    features = ddf[param['feature_variables']]
    #features.persist()
    k = int(param['options']['params']['k'])
    model['dask_kmeans'] = dask_ml.cluster.KMeans(n_clusters=k, init_max_iter=2, oversampling_factor=10)
    model['dask_kmeans'].fit(features)
    return model







    
# In[20]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    #ddf = dd.from_pandas(df[param['feature_variables']], npartitions=4)
    prediction = model['dask_kmeans'].labels_
    #y_hat = prediction.to_dask_dataframe()
    result = pd.DataFrame(prediction.compute())
    model['dask_client'].close()
    return result







    
# In[11]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    # TODO
    # model['dask_kmeans'].save_model(MODEL_DIRECTORY + name + '.json')
    return model







    
# In[13]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    # TODO
    model = {}
    return model







    
# In[15]:


# return a model summary
def summary(model=None):
    returns = {"model": model}
    return returns







