#!/usr/bin/env python
# coding: utf-8


    
# In[15]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import urllib
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from pymilvus import MilvusClient
from pinecone import Pinecone, ServerlessSpec
from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, AlloyDBDocumentStore, AlloyDBIndexStore
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"









    
# In[10]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    return model







    
# In[12]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[14]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    # Manager task includes list_collections, delete_collection, show_schema and show_rows
    task = param['options']['params']['task']
    try:
        collection_name = param['options']['params']['collection_name']
    except:
        collection_name = 'default_collection'

    try:
        vectordb_service = param['options']['params']['vectordb_service']
    except:
        vectordb_service = 'milvus'
    if vectordb_service == 'milvus':
        uri = json.loads(os.environ['llm_config'])['vector_db']['milvus'][0]['uri']
        token = json.loads(os.environ['llm_config'])['vector_db']['milvus'][0]['token']
        host = urllib.parse.urlparse(uri).hostname
        port = urllib.parse.urlparse(uri).port
        client = MilvusClient(
            uri=uri,
            token=token
        )
        
        connections.connect("default", host=host, port=port)
        
        if task == "general":
            collection_list = client.list_collections()
            schemas = []
            rows = []
            for clt in collection_list:
                collection = Collection(clt)
                m1 = str([item.name for item in collection.schema.fields])
                schemas.append(m1)
                rows.append(collection.num_entities)
            cols = {"Collections": collection_list, "Fields": schemas, "Number_of_rows": rows}
                
        elif task == "list_collections":
            collection_list = "|".join(client.list_collections())
            cols = {"Collections": [collection_list], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": ["No result"]}
        elif task == "delete_collection":
            utility.drop_collection(collection_name)
            m = f'Deleted collection {collection_name}'
            cols = {"Collections": ["No result"], "Message": [m], "Schema": ["No result"], "Number_of_rows": ["No result"]}
        elif task == "show_schema":
            try:
                collection = Collection(collection_name)
                s = str(collection.schema.fields)
            except:
                s = ''
            cols = {"Collections": ["No result"], "Message": ["No result"], "Schema": [s], "Number_of_rows": ["No result"]}
        elif task == "show_rows":
            try:
                collection = Collection(collection_name)
                n = str(collection.num_entities)
            except:
                n = ''
            cols = {"Collections": ["No result"], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": [n]}
        else:
            cols = {"Collections": ["No result"], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": ["No result"]}
    elif vectordb_service == 'pinecone':
        api_key = json.loads(os.environ['llm_config'])['vector_db']['pinecone'][0]['api_key']
        pc = Pinecone(api_key=api_key)
            
        if task == "list_collections":
            try:
                collection_list = []
                index_list = pc.list_indexes()['indexes']
                for item in index_list:
                    collection_list.append(item['name'])
                collection_list = "|".join(collection_list)
            except:
                collection_list = ""
            cols = {"Collections": [collection_list], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": ["No result"]}
                
        else:
            cols = {"Collections": ["No result"], "Message": ["Task not supported for pinecone."], "Schema": ["No result"], "Number_of_rows": ["No result"]}

    else:
        if task == "list_collections":
             cols = {"Collections": ["No result"], "Message": ["Task not supported for alloyDB. Please specify collection names manually."], "Schema": ["No result"], "Number_of_rows": ["No result"]}   
        else:
            cols = {"Collections": ["No result"], "Message": ["Task not supported for alloyDB."], "Schema": ["No result"], "Number_of_rows": ["No result"]}

        
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

def compute(model,df,param):
    # Manager task includes list_collections, delete_collection, show_schema and show_rows
    task = param['options']['params']['task']
    try:
        collection_name = param['options']['params']['collection_name']
    except:
        collection_name = 'default_collection'

    try:
        vectordb_service = param['options']['params']['vectordb_service']
    except:
        vectordb_service = 'milvus'
    if vectordb_service == 'milvus':
        uri = json.loads(os.environ['llm_config'])['vector_db']['milvus'][0]['uri']
        token = json.loads(os.environ['llm_config'])['vector_db']['milvus'][0]['token']
        host = urllib.parse.urlparse(uri).hostname
        port = urllib.parse.urlparse(uri).port
        client = MilvusClient(
            uri=uri,
            token=token
        )
        
        connections.connect("default", host=host, port=port)
        
        if task == "general":
            collection_list = client.list_collections()
            schemas = []
            rows = []
            for clt in collection_list:
                collection = Collection(clt)
                m1 = str([item.name for item in collection.schema.fields])
                schemas.append(m1)
                rows.append(collection.num_entities)
            cols = {"Collections": collection_list, "Fields": schemas, "Number_of_rows": rows}
                
        elif task == "list_collections":
            collection_list = "|".join(client.list_collections())
            cols = {"Collections": [collection_list], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": ["No result"]}
        elif task == "delete_collection":
            utility.drop_collection(collection_name)
            m = f'Deleted collection {collection_name}'
            cols = {"Collections": ["No result"], "Message": [m], "Schema": ["No result"], "Number_of_rows": ["No result"]}
        elif task == "show_schema":
            try:
                collection = Collection(collection_name)
                s = str(collection.schema.fields)
            except:
                s = ''
            cols = {"Collections": ["No result"], "Message": ["No result"], "Schema": [s], "Number_of_rows": ["No result"]}
        elif task == "show_rows":
            try:
                collection = Collection(collection_name)
                n = str(collection.num_entities)
            except:
                n = ''
            cols = {"Collections": ["No result"], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": [n]}
        else:
            cols = {"Collections": ["No result"], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": ["No result"]}
    elif vectordb_service == 'pinecone':
        api_key = json.loads(os.environ['llm_config'])['vector_db']['pinecone'][0]['api_key']
        pc = Pinecone(api_key=api_key)
            
        if task == "list_collections":
            try:
                collection_list = []
                index_list = pc.list_indexes()['indexes']
                for item in index_list:
                    collection_list.append(item['name'])
                collection_list = "|".join(collection_list)
            except:
                collection_list = ""
            cols = {"Collections": [collection_list], "Message": ["No result"], "Schema": ["No result"], "Number_of_rows": ["No result"]}
                
        else:
            cols = {"Collections": ["No result"], "Message": ["Task not supported for pinecone."], "Schema": ["No result"], "Number_of_rows": ["No result"]}

    else:
        if task == "list_collections":
             cols = {"Collections": ["No result"], "Message": ["Task not supported for alloyDB. Please specify collection names manually."], "Schema": ["No result"], "Number_of_rows": ["No result"]}   
        else:
            cols = {"Collections": ["No result"], "Message": ["Task not supported for alloyDB."], "Schema": ["No result"], "Number_of_rows": ["No result"]}

        
    result = pd.DataFrame(data=cols)
    return result

















