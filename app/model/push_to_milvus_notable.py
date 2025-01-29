#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os

import time

import pymilvus

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

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











    
# In[20]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}

    try:
        n_dims=int(param['options']['params']['n_dims'])
    except:
        n_dims=8
        
    pk_type=DataType.VARCHAR        
    embedding_type=DataType.FLOAT_VECTOR
    
    try:
        collection_name=param['options']['params']['collection_name']
    except:
        collection_name="default_collection"
        
    print("start connecting to Milvus")
    # this hostname may need changing to a specific local docker network ip address depending on docker configuration
    connections.connect("default", host="milvus-standalone", port="19530")

    collection_exists = utility.has_collection(collection_name)

    # add schema ['search_name', 'src', 'dest', 'mitre_id', 'info_min_time']
    fields = [
        FieldSchema(name="_key", is_primary=True, auto_id=True, dtype=DataType.INT64),
        FieldSchema(name="embeddings", dtype=embedding_type, dim=n_dims),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="search_name", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="src", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="dest", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="mitre_id", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="info_min_time", dtype=DataType.FLOAT)
    ]

    schema = CollectionSchema(fields, f"dsdl schema for {collection_name}")
    
    if collection_exists:
        print(f"The collection {collection_name} already exists")
        collection = Collection(collection_name)
        collection.load()
    else:
        print(f"The collection {collection_name} does not exist")
        print(f"creating new collection: {collection_name}")
        collection = Collection(collection_name, schema, consistency_level="Strong")
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        collection.create_index("embeddings", index)
    
    model['collection']=collection
    model['collection_name']=collection_name

    return model







    
# In[28]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    df=df.copy()
    label_field_name=param['options']['params']['label_field_name']
    label_column = df[label_field_name].astype(str)
    # add schema ['search_name', 'src', 'dest', 'mitre_id', 'info_min_time']
    search_name = df['search_name'].astype(str)
    src = df['src'].astype(str)
    dest = df['dest'].astype(str)
    mitre_id = df['annotations.mitre_attack.mitre_tactic_id'].astype(str)
    info_min_time = df['info_min_time'].astype(float)
    df.drop(label_field_name, axis=1, inplace=True)
    df.drop(['search_name', 'src', 'dest', 'annotations.mitre_attack.mitre_tactic_id', 'info_min_time'], axis=1, inplace=True)
    df_list=df.values.tolist()
    data=[ df_list, label_column.tolist(), search_name.tolist(), src.tolist(), dest.tolist(), mitre_id.tolist(), info_min_time.tolist()]
    model['collection'].insert(data)    
    info = {"message": f"inserted data to collection {model['collection_name']}"}
    return df







    
# In[30]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    return df







    
# In[1]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    return model





    
# In[2]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    return model





    
# In[17]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

















