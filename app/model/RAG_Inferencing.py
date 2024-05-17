#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, ServiceContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import textwrap

# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"











    
# In[6]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param





    
# In[10]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    return model





    
# In[19]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    info = {"message": "model trained"}
    return info









    
# In[21]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    # Example: 'all-MiniLM-L6-v2'
    X = df["query"].values.tolist()
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
        embedder_dimension = param['options']['params']['embedder_dimension']
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        embedder_dimension = 384
        collection_name = "default-doc-collection"

    # TODO: Check existence of collection

    model="llama3"
    url = "http://ollama:11434"
    llm = Ollama(model=model, base_url=url)
    transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=transformer_embedder, chunk_size=1024
    )
    vector_store = MilvusVectorStore(uri="http://milvus-standalone:19530", token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
    index = VectorStoreIndex.from_vector_store(
           vector_store=vector_store, service_context=service_context
        )
    query_engine = index.as_query_engine()
    l = []
    for i in range(len(X)):
        r = query_engine.query(X[i])
        l.append(r.response)
    

    cols = {"Response": l}
    result = pd.DataFrame(data=cols)
    return result







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[18]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

















