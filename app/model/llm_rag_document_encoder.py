#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, ServiceContext, Settings
from llama_index.readers.file import DocxReader, CSVReader, PDFReader, PptxReader, XMLReader, IPYNBReader 
from llama_index.vector_stores.milvus import MilvusVectorStore
import textwrap
from app.model.llm_utils import create_llm, create_embedding_model, create_vector_db
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







    
# In[22]:


def apply(model,df,param):
    # Datapath Example: '/srv/notebooks/data/splunk_doc/'
    try:
        data_path = param['options']['params']['data_path'].strip('\"')
    except:
        data_path = None
        print("No file path specified.")

    try:
        vec_service = param['options']['params']['vectordb_service'].strip('\"')
        print(f"Using {vec_service} vector database service")
    except:
        vec_service = "milvus"
        print("Using default Milvus vector database service")
    
    try:
        service = param['options']['params']['embedder_service'].strip('\"')
        print(f"Using {service} embedding service")
    except:
        service = "huggingface"
        print("Using default Huggingface embedding service")
        
    try:
        embedder_dimension = int(param['options']['params']['embedder_dimension'])
    except:
        embedder_dimension = None
        print("embedder_dimension not specified.")
        
    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        collection_name = "default-doc-collection"
        print("collection_name not specified. Use default-doc-collection by default.")
        
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = None
        print("embedder_name not specified.")

    try:
        use_local = int(param['options']['params']['use_local'])
    except:
        use_local = 0
        print("Not using local model.")
        
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
    except Exception as e:
        documents = None
        message = f"ERROR in directory loading: {e} "

    try:
        embedder, output_dims, m = create_embedding_model(service=service, model=embedder_name, use_local=use_local)
        if embedder is not None:
            print(m)
        else:
            message = f"ERROR in embedding model loading: {m}. "
        if output_dims:
            embedder_dimension = output_dims
    except Exception as e:
        embedder = None
        message = f"ERROR in embedding model loading: {e}. Check if the model name is correct. If you selected Yes for use local embedder, make sure you have pulled the embedding model to local."

    if (documents is not None) & (embedder is not None):
        try:
            # Replacing service context in legacy llama-index
            Settings.llm = None
            Settings.embed_model = embedder
            Settings.chunk_size = 1024
            # Creating vectorDB object
            vector_store, v_m = create_vector_db(service=vec_service, collection_name=collection_name, dim=embedder_dimension)
            if vector_store is None:
                cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [f"Failed at creating vector database: {v_m}"]}
                result = pd.DataFrame(data=cols)
                return result
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Index document data
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            # Prepare output dataframe
            embedder_info = [m]
            vector_store = [str(vector_store)]
            document = []
            for d in documents:
                document.append(str(d.metadata['file_path']))
            document = str(list(dict.fromkeys(document)))
            cols = {"Embedder_Info": embedder_info, "Vector_Store_Info": vector_store, "Documents_Info": [document], "Message": ["Success"]}
        except Exception as e:
            message = f"ERROR in encoding: {e}."
            cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
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





    
# In[25]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns
def compute(model,df,param):
    # Datapath Example: '/srv/notebooks/data/splunk_doc/'
    try:
        data_path = param['options']['params']['data_path'].strip('\"')
    except:
        data_path = None
        print("No file path specified.")

    try:
        vec_service = param['options']['params']['vectordb_service'].strip('\"')
        print(f"Using {vec_service} vector database service")
    except:
        vec_service = "milvus"
        print("Using default Milvus vector database service")
    
    try:
        service = param['options']['params']['embedder_service'].strip('\"')
        print(f"Using {service} embedding service")
    except:
        service = "huggingface"
        print("Using default Huggingface embedding service")
        
    try:
        embedder_dimension = int(param['options']['params']['embedder_dimension'])
    except:
        embedder_dimension = None
        print("embedder_dimension not specified.")
        
    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        collection_name = "default-doc-collection"
        print("collection_name not specified. Use default-doc-collection by default.")
        
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = None
        print("embedder_name not specified.")

    try:
        use_local = int(param['options']['params']['use_local'])
    except:
        use_local = 0
        print("Not using local model.")
        
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
    except Exception as e:
        documents = None
        message = f"ERROR in directory loading: {e} "

    try:
        embedder, output_dims, m = create_embedding_model(service=service, model=embedder_name, use_local=use_local)
        if embedder is not None:
            print(m)
        else:
            message = f"ERROR in embedding model loading: {m}. "
        if output_dims:
            embedder_dimension = output_dims
    except Exception as e:
        embedder = None
        message = f"ERROR in embedding model loading: {e}. Check if the model name is correct. If you selected Yes for use local embedder, make sure you have pulled the embedding model to local."

    if (documents is not None) & (embedder is not None):
        try:
            # Replacing service context in legacy llama-index
            Settings.llm = None
            Settings.embed_model = embedder
            Settings.chunk_size = 1024
            # Creating vectorDB object
            vector_store, v_m = create_vector_db(service=vec_service, collection_name=collection_name, dim=embedder_dimension)
            if vector_store is None:
                cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [f"Failed at creating vector database: {v_m}"]}
                result = pd.DataFrame(data=cols)
                return result
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Index document data
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            # Prepare output dataframe
            embedder_info = [m]
            vector_store = [str(vector_store)]
            document = []
            for d in documents:
                document.append(str(d.metadata['file_path']))
            document = str(list(dict.fromkeys(document)))
            cols = {"Embedder_Info": embedder_info, "Vector_Store_Info": vector_store, "Documents_Info": [document], "Message": ["Success"]}
        except Exception as e:
            message = f"ERROR in encoding: {e}."
            cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
    else:
        cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
    result = pd.DataFrame(data=cols)
    return result







