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
from llama_index.readers.file import DocxReader, CSVReader, PDFReader, PptxReader, XMLReader, IPYNBReader 
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import textwrap

# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
MILVUS_ENDPOINT = "http://milvus-standalone:19530"









    
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







    
# In[1]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    # Datapath Example: '/srv/notebooks/data/splunk_doc/'
    data_path = param['options']['params']['data_path'].strip('\"')
    use_local= int(param['options']['params']['use_local'])
    # Embedder Example: 'all-MiniLM-L6-v2'
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
        embedder_dimension = param['options']['params']['embedder_dimension']
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        embedder_dimension = 384
        collection_name = "default-doc-collection"
    # Dimension checking for default embedders
    if embedder_name == 'intfloat/multilingual-e5-large':
        embedder_dimension = 1024
    elif embedder_name == 'all-MiniLM-L6-v2':
        embedder_dimension = 384
    else:
        embedder_dimension = embedder_dimension
    # Using local embedder checkpoints
    if use_local:
        embedder_name = f'/srv/app/model/data/{embedder_name}'
        print("Using local embedding model checkpoints")
        
    # To support pptx files, huggingface extractor needs to be downloaded. Skipping support for this version
    # Special parser for CSV data
    parser = CSVReader()
    file_extractor = {".csv": parser}
    try:
        # Create document dataloader - recursively find data from sub-directories
        # Add desired file extensions in required_exts. For example: required_exts=[".csv", ".xml", ".pdf", ".docx", ".ipynb"]
        documents = SimpleDirectoryReader(
            input_dir=data_path, recursive=True, file_extractor=file_extractor, required_exts=[".ipynb", ".csv", ".xml", ".pdf", ".txt", ".docx"]
        ).load_data()
    except:
        documents = None
        message = "ERROR: No data in the directory specified. Check if the directory exists and contains files."
    # Create Transformers embedding model 
    ## TODO: add local loading option
    try:
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
        print(f'Loaded embedder from {embedder_name}')
    except:
        transformer_embedder = None
        message = "ERROR: embedding model is not loaded. Check if the model name is correct. For local loading, check if the path exists"

    if (documents is not None) & (transformer_embedder is not None):
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=transformer_embedder, chunk_size=1024
        )
        vector_store = MilvusVectorStore(uri=MILVUS_ENDPOINT, token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Index document data
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=service_context
        )
    
        # Prepare output dataframe
        embedder = [str(transformer_embedder)]
        vector_store = [str(vector_store)]
        document = []
        for d in documents:
            document.append(str(d.metadata['file_path']))
        document = str(list(dict.fromkeys(document)))
        cols = {"Embedder_Info": embedder, "Vector_Store_Info": vector_store, "Documents_Info": [document], "Message": ["Success"]}
    else:
        cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
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
    # Datapath Example: '/srv/notebooks/data/splunk_doc/'
    data_path = param['params']['data_path'].strip('\"')
    use_local= int(param['params']['use_local'])
    # Embedder Example: 'all-MiniLM-L6-v2'
    try:
        embedder_name = param['params']['embedder_name'].strip('\"')
        embedder_dimension = param['params']['embedder_dimension']
        collection_name = param['params']['collection_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        embedder_dimension = 384
        collection_name = "default-doc-collection"
    # Dimension checking for default embedders
    if embedder_name == 'intfloat/multilingual-e5-large':
        embedder_dimension = 1024
    elif embedder_name == 'all-MiniLM-L6-v2':
        embedder_dimension = 384
    else:
        embedder_dimension = embedder_dimension
    # Using local embedder checkpoints
    if use_local:
        embedder_name = f'/srv/app/model/data/{embedder_name}'
        print("Using local embedding model checkpoints")
        
    # To support pptx files, huggingface extractor needs to be downloaded. Skipping support for this version
    # Special parser for CSV data
    parser = CSVReader()
    file_extractor = {".csv": parser}
    try:
        # Create document dataloader - recursively find data from sub-directories
        # Add desired file extensions in required_exts. For example: required_exts=[".csv", ".xml", ".pdf", ".docx", ".ipynb"]
        documents = SimpleDirectoryReader(
            input_dir=data_path, recursive=True, file_extractor=file_extractor, required_exts=[".ipynb", ".csv", ".xml", ".pdf", ".txt", ".docx"]
        ).load_data()
    except:
        documents = None
        message = "ERROR: No data in the directory specified. Check if the directory exists and contains files."
    # Create Transformers embedding model 
    ## TODO: add local loading option
    try:
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
        print(f'Loaded embedder from {embedder_name}')
    except:
        transformer_embedder = None
        message = "ERROR: embedding model is not loaded. Check if the model name is correct. For local loading, check if the path exists"

    if (documents is not None) & (transformer_embedder is not None):
        print("Start encoding")
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=transformer_embedder, chunk_size=1024
        )
        vector_store = MilvusVectorStore(uri=MILVUS_ENDPOINT, token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
        print("Vector store ok")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Index document data
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=service_context
        )
        print("Index store ok")
        # Prepare output dataframe
        embedder = str(transformer_embedder)
        vector_store = str(vector_store)
        document = []
        for d in documents:
            document.append(str(d.metadata['file_path']))
        print("Finished encoding")
        document = str(list(dict.fromkeys(document)))
        cols = {"Embedder_Info": embedder, "Vector_Store_Info": vector_store, "Documents_Info": document, "Message": "Success"}
    else:
        cols = {"Embedder_Info": "No Result", "Vector_Store_Info": "No Result", "Documents_Info": "No Result", "Message": message}
    result = [cols]
    return result







