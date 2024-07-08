#!/usr/bin/env python
# coding: utf-8


    
# In[18]:


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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
MILVUS_ENDPOINT = "http://milvus-standalone:19530"







    
# In[19]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[14]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}   
    pk_type=DataType.VARCHAR        
    embedding_type=DataType.FLOAT_VECTOR
    # Dimensionality setting of collection
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
    # Dimension checking for default embedders
    if embedder_name == 'intfloat/multilingual-e5-large':
        n_dims = 1024
    elif embedder_name == 'all-MiniLM-L6-v2':
        n_dims = 384
    else:
        try:
            n_dims=int(param['options']['params']['embedder_dimension'])
        except:
            n_dims=384
    
    
    # Collection name setting   
    try:
        collection_name=param['options']['params']['collection_name'].strip('\"')
    except:
        collection_name="default_collection"
    # Schema setting
    try:
        schema_fields=df.columns.tolist()
        schema_fields.remove(param['options']['params']['label_field_name'])
    except:
        schema_fields=[]
        
    print("start connecting to Milvus")
    try:
        # this hostname may need changing to a specific local docker network ip address depending on docker configuration
        connections.connect("default", host="milvus-standalone", port="19530")
        collection_exists = utility.has_collection(collection_name)
        
        # Basic schema setting
        fields = [
            FieldSchema(name="_key", is_primary=True, auto_id=True, dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=embedding_type, dim=n_dims),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=15000),
        ]
        # Additional schema setting
        if len(schema_fields) != 0: 
            for i in range(len(schema_fields)):
                fields.append(FieldSchema(name=schema_fields[i], dtype=DataType.VARCHAR, max_length=1000))
        # Create schema
        
        schema = CollectionSchema(fields, f"dsdl schema for {collection_name}")
        print(fields)
        
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
    except:
        collection = None
    
    model['collection']=collection
    model['collection_name']=collection_name

    return model







    
# In[18]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    
    return df







    
# In[16]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    if model['collection'] is not None:
        use_local= int(param['options']['params']['use_local'])
        try:
            embedder_name = param['options']['params']['embedder_name'].strip('\"')
        except:
            embedder_name = 'all-MiniLM-L6-v2'
        if use_local:
            embedder_name = f'/srv/app/model/data/{embedder_name}'
            print("Using local embedding model checkpoints")  
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)

        try:
            df=df.copy()
            label_field_name=param['options']['params']['label_field_name']
            label_column = df[label_field_name].astype(str)
        
            text_column = label_column.tolist()
            vector_column = []
            for text in text_column:
                vector_column.append(transformer_embedder.get_text_embedding(text))
            data=[vector_column, label_column.tolist()]
        except:
            data = None
            m = "Failed. Could not vectorize dataframe. Check your field name."
            
        try:
            schema_fields=df.columns.tolist()
            schema_fields.remove(label_field_name)
        except:
            schema_fields=[]
        if data is not None:
            if len(schema_fields) != 0:
                for i in range(len(schema_fields)):  
                    data.append(df[schema_fields[i]].astype(str).tolist())
            # Cap at 16MB for each insertion, 1/4 of the 64MB limit
            data_limit = 16000000
            try:
                n_dims=int(param['options']['params']['embedder_dimension'])
            except:
                n_dims=384
            print(f"Size of data is {len(data[0])}")
            num_vectors = int(data_limit / (n_dims * 4))
            print(f"Batch size is {num_vectors}")
            num_sublists = len(data[0]) // num_vectors
            print(f"Number of batches is {num_sublists}")
            # Initialize the sublists
            sublists = [[] for _ in range(num_sublists)]
            # Iterate over each row in the data
            for row in data:
                for i in range(num_sublists):
                    sublists[i].append(row[i * num_vectors:(i + 1) * num_vectors])
            try:
                for sub_data in sublists:
                    model['collection'].insert(sub_data, timeout=None)
                    print(f"Inserted data batch with length {len(sub_data[0])}")
                m = "Success"
            except:
                m = "Failed. Could not insert data to collection."
    else:
        m = "Failed. Could not create collection. Check collection naming."
    df['message'] = [m]*df.shape[0]
    return df['message']







    
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

def compute(model,df,param):
    model = {}   
    pk_type=DataType.VARCHAR        
    embedding_type=DataType.FLOAT_VECTOR
    # Dimensionality setting of collection
    try:
        embedder_name = param['params']['embedder_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
    # Dimension checking for default embedders
    if embedder_name == 'intfloat/multilingual-e5-large':
        n_dims = 1024
    elif embedder_name == 'all-MiniLM-L6-v2':
        n_dims = 384
    else:
        try:
            n_dims=int(param['params']['embedder_dimension'])
        except:
            n_dims=384
    
    
    # Collection name setting   
    try:
        collection_name=param['params']['collection_name'].strip('\"')
    except:
        collection_name="default_collection"
    # Schema setting
    try:
        schema_fields=param['fieldnames']
        schema_fields.remove(param['params']['label_field_name'])
    except:
        schema_fields=[]

    print(schema_fields)
        
    print("start connecting to Milvus")
    try:
        # this hostname may need changing to a specific local docker network ip address depending on docker configuration
        connections.connect("default", host="milvus-standalone", port="19530")
        collection_exists = utility.has_collection(collection_name)
        
        # Basic schema setting
        fields = [
            FieldSchema(name="_key", is_primary=True, auto_id=True, dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=embedding_type, dim=n_dims),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=15000),
        ]
        # Additional schema setting
        if len(schema_fields) != 0: 
            for i in range(len(schema_fields)):
                fields.append(FieldSchema(name=schema_fields[i], dtype=DataType.VARCHAR, max_length=1000))
        # Create schema
        
        schema = CollectionSchema(fields, f"dsdl schema for {collection_name}")
        print(fields)
        
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
    except:
        collection = None
    
    model['collection']=collection
    model['collection_name']=collection_name

    if model['collection'] is not None:
        use_local= int(param['params']['use_local'])
        try:
            embedder_name = param['params']['embedder_name'].strip('\"')
        except:
            embedder_name = 'all-MiniLM-L6-v2'
        if use_local:
            embedder_name = f'/srv/app/model/data/{embedder_name}'
            print("Using local embedding model checkpoints")  
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)

        try:
            label_field_name=param['params']['label_field_name']
            print(label_field_name)
            texts = []
            vectors = []
            for i in range(len(df)):
                texts.append(df[i][label_field_name])
                vectors.append(transformer_embedder.get_text_embedding(df[i][label_field_name]))
            data=[vectors, texts]
        except:
            data = None
            m = {"Message": "Failed. Could not vectorize dataframe. Check your field name."}
            print(m)
            
        if data is not None:
            if len(schema_fields) != 0:
                for field in schema_fields:  
                    l = []
                    for i in range(len(df)):
                        l.append(df[i][field])
                    data.append(l)
            data_limit = 16000000
            print(f"Size of data is {len(data[0])}")
            num_vectors = int(data_limit / (n_dims * 4))
            print(f"Batch size is {num_vectors}")
            num_sublists = len(data[0]) // num_vectors
            print(f"Number of batches is {num_sublists}")
            # Initialize the sublists
            sublists = [[] for _ in range(num_sublists)]
            for row in data:
                for i in range(num_sublists):
                    sublists[i].append(row[i * num_vectors:(i + 1) * num_vectors])
            try:
                for sub_data in sublists:
                    model['collection'].insert(sub_data, timeout=None)
                    print(f"Inserted data batch with length {len(sub_data[0])}")
                m = {"Message": "Success"}
                print(m)
            except:
                m = {"Message": "Failed. Too much data to insert at once."}
                print(m)
    else:
        m = {"Message": "Failed. Could not create collection. Check collection naming."}
        print(m)
    cols =[]
    for _ in range(len(df)):
        cols.append(m)
    return cols















