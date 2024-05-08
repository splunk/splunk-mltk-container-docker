#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import networkx as nx
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







    
# In[7]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = nx.Graph()
    return model







    
# In[10]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):

    model.clear()
    src_dest_name = param['feature_variables']
    dfg = df[src_dest_name]
    for index, row in dfg.iterrows():
        model.add_edge(row[src_dest_name[0]], row[src_dest_name[1]]) #, value=row['value'])
    return model







    
# In[12]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    src_dest_name = param['feature_variables']
    algos = param['options']['params']['compute'].lstrip("\"").rstrip("\"").lower().split(',')
    outputcolumns = []
    for algo in algos:
        if algo=='degree_centrality':
            cents = nx.algorithms.centrality.degree_centrality(model)
            outputcolumns.append(algo)
        elif algo=='betweenness_centrality':
            cents = nx.algorithms.centrality.betweenness_centrality(model)
            outputcolumns.append(algo)
        elif algo=='eigenvector_centrality':
            cents = nx.algorithms.centrality.eigenvector_centrality(model, max_iter=200)
            outputcolumns.append(algo)
        elif algo=='cluster_coefficient':
            cents = nx.algorithms.cluster.clustering(model)
            outputcolumns.append(algo)
        else:
            continue
        degs = pd.DataFrame(list(cents.items()), columns=[src_dest_name[0], algo])
        df = df.join(degs.set_index(src_dest_name[0]), on=src_dest_name[0])
    return df[outputcolumns]







    
# In[14]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    # with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
    #    json.dump(model, file)
    return model





    
# In[15]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = init(None,None)
    # with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
    #    model = json.load(file)
    return model





    
# In[16]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__, "networkx": nx.__version__} }
    return returns





