#!/usr/bin/env python
# coding: utf-8


    
# In[13]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import networkx as nx
import graphistry
# please use your graphistry credentials to use their services.
# security note: your graph data is sent to graphistry hub, so please ensure all your data security and compliance is in sync with this operation
graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="username", password="XXXXXXXXX")    

# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[15]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param





















    
# In[31]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = graphistry.edges(df).bind(source='src_ip', destination='dst_ip')
    return model







    
# In[34]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    model = graphistry.edges(df).bind(source='src_ip', destination='dst_ip')
    return model







    
# In[37]:


# apply the model
# returns the calculated results
def apply(model,df,param):
    # example to utilize graphistry functions to derive insights from the graph and return to Splunk
    topo = model.get_topological_levels()
    return topo._nodes







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    # with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
    #    json.dump(model, file)
    return model





    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = init(None,None)
    # with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
    #    model = json.load(file)
    return model





    
# In[40]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__, "graphistry": graphistry.__version__} }
    return returns







