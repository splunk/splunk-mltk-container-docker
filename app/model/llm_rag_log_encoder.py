#!/usr/bin/env python
# coding: utf-8


    
# In[3]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import time
from llama_index.core import VectorStoreIndex, Document, StorageContext, ServiceContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from app.model.llm_utils import create_llm, create_embedding_model, create_vector_db
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[2]:


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
    return model







    
# In[18]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):  
    return df







    
# In[8]:


def apply(model,df,param):
    result_dict = {"embedder_Info": ["No Result"], "vector_Store_Info": ["No Result"], "message": [""]}
    
    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        data = None
        result_dict["message"].append("Please specify a collection_name parameter for the vectorDB collection.")
        return pd.DataFrame(data=result_dict)

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
        use_local= int(param['options']['params']['use_local'])
    except:
        use_local=0

    try:
        label_field_name=param['options']['params']['label_field_name']
    except:
        data = None
        result_dict["message"] = ["Failed to preprocess data. Please specify a label_field_name parameter for the field to encode."]
        return pd.DataFrame(data=result_dict)

    try:
        embedder_dimension = int(param['options']['params']['embedder_dimension'])
    except:
        embedder_dimension = None
        print("embedder_dimension not specified.")
    
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

    try:
        embedder, output_dims, m = create_embedding_model(service=service, model=embedder_name, use_local=use_local)

        if embedder is not None:
            result_dict["embedder_Info"] = [m]
        else:
            message = f"ERROR in embedding model loading: {m}. "
            result_dict["message"] = [m]
            return pd.DataFrame(data=result_dict)
        if output_dims:
            embedder_dimension = output_dims       
    except Exception as e:
        m = f"Failed to initiate embedding model. ERROR: {e}"
        result_dict["message"] = [m]
        return pd.DataFrame(data=result_dict)

    try:
        df=df.copy()
        text_df = df[label_field_name].astype(str).tolist()
        meta_df = df.drop(label_field_name, axis=1).astype(str)

        if meta_df.empty:
            documents = [Document(text=text) for text in text_df]            
        else:
            meta_records = meta_df.to_dict('records')
            meta_fields = meta_df.columns.tolist()
            documents = [Document(text=text, metadata=meta, excluded_embed_metadata_keys=meta_fields, excluded_llm_metadata_keys=meta_fields) for text, meta in zip(text_df, meta_records)]

        doc_count = len(documents)
    except KeyError as e:
        data = None
        result_dict["message"] = f"Failed at data preprocessing. Could not find label_field_name {label_field_name} in data. ERROR:{e}"
        return pd.DataFrame(data=result_dict)
    except Exception as e:
        data = None
        result_dict["message"] = f"Failed at data preprocessing. ERROR:{e}"
        return pd.DataFrame(data=result_dict)

    if (documents is None) or (embedder is None):
        result_dict["message"] = f"Failed to load input data or embedding model. Input data:{documents}, Embedding model:{embedder}"
        return pd.DataFrame(data=result_dict)
        
    try:
        Settings.llm = None
        Settings.embed_model = embedder
        # similarity_metric set to default value: IP (inner-product)
        vector_store, v_m = create_vector_db(service=vec_service, collection_name=collection_name, dim=embedder_dimension)
        if vector_store is None:
            result_dict["message"] = f"Failed at creating vectordb object. ERROR:{v_m}"
            return pd.DataFrame(data=result_dict)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        result_dict["message"] = "Success"
        result_dict["embedder_Info"] = [m]
        result_dict["vector_Store_Info"] = [str(vector_store)]

    except Exception as e:
        result_dict["message"] = f"Failed at vectorization. ERROR:{e}"
        return pd.DataFrame(data=result_dict)
    
    return pd.DataFrame(data=result_dict)







    
# In[1]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    return model





    
# In[2]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    return model





    
# In[46]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

def compute(model,df,param):
    result_dict = {"embedder_Info": ["No Result"], "vector_Store_Info": ["No Result"], "message": [""]}
    
    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        data = None
        result_dict["message"].append("Please specify a collection_name parameter for the vectorDB collection.")
        return pd.DataFrame(data=result_dict)

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
        use_local= int(param['options']['params']['use_local'])
    except:
        use_local=0

    try:
        label_field_name=param['options']['params']['label_field_name']
    except:
        data = None
        result_dict["message"] = ["Failed to preprocess data. Please specify a label_field_name parameter for the field to encode."]
        return pd.DataFrame(data=result_dict)

    try:
        embedder_dimension = int(param['options']['params']['embedder_dimension'])
    except:
        embedder_dimension = None
        print("embedder_dimension not specified.")
    
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

    try:
        embedder, output_dims, m = create_embedding_model(service=service, model=embedder_name, use_local=use_local)

        if embedder is not None:
            result_dict["embedder_Info"] = [m]
        else:
            message = f"ERROR in embedding model loading: {m}. "
            result_dict["message"] = [m]
            return pd.DataFrame(data=result_dict)
        if output_dims:
            embedder_dimension = output_dims       
    except Exception as e:
        m = f"Failed to initiate embedding model. ERROR: {e}"
        result_dict["message"] = [m]
        return pd.DataFrame(data=result_dict)

    try:
        df=df.copy()
        text_df = df[label_field_name].astype(str).tolist()
        meta_df = df.drop(label_field_name, axis=1).astype(str)

        if meta_df.empty:
            documents = [Document(text=text) for text in text_df]            
        else:
            meta_records = meta_df.to_dict('records')
            meta_fields = meta_df.columns.tolist()
            documents = [Document(text=text, metadata=meta, excluded_embed_metadata_keys=meta_fields, excluded_llm_metadata_keys=meta_fields) for text, meta in zip(text_df, meta_records)]

        doc_count = len(documents)
    except KeyError as e:
        data = None
        result_dict["message"] = f"Failed at data preprocessing. Could not find label_field_name {label_field_name} in data. ERROR:{e}"
        return pd.DataFrame(data=result_dict)
    except Exception as e:
        data = None
        result_dict["message"] = f"Failed at data preprocessing. ERROR:{e}"
        return pd.DataFrame(data=result_dict)

    if (documents is None) or (embedder is None):
        result_dict["message"] = f"Failed to load input data or embedding model. Input data:{documents}, Embedding model:{embedder}"
        return pd.DataFrame(data=result_dict)
        
    try:
        Settings.llm = None
        Settings.embed_model = embedder
        # similarity_metric set to default value: IP (inner-product)
        vector_store, v_m = create_vector_db(service=vec_service, collection_name=collection_name, dim=embedder_dimension)
        if vector_store is None:
            result_dict["message"] = f"Failed at creating vectordb object. ERROR:{v_m}"
            return pd.DataFrame(data=result_dict)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        result_dict["message"] = "Success"
        result_dict["embedder_Info"] = [m]
        result_dict["vector_Store_Info"] = [str(vector_store)]

    except Exception as e:
        result_dict["message"] = f"Failed at vectorization. ERROR:{e}"
        return pd.DataFrame(data=result_dict)
    
    return pd.DataFrame(data=result_dict)

















