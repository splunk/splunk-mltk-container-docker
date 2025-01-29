#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import requests
import os
from sentence_transformers import SentenceTransformer
# ...
# global constants
ollama_url = "http://ollama:11434"
MODEL_DIRECTORY = "/srv/app/model/data/"





    
# In[2]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[ ]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    return model







    
# In[ ]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[ ]:


def apply(model,df,param):
    manager = param['options']['params']['task'].strip('\"')
    try:
        model_type = param['options']['params']['model_type'].strip('\"')
    except:
        model_type = "LLM"
    
    if model_type == "embedder_model":
        # embedder model
        if manager == "pull":
            model_name = param['options']['params']['model_name'].strip('\"').strip('\'')
            if model_name == "all-MiniLM-L6-v2" or model_name == "intfloat/multilingual-e5-large":
                modelPath = os.path.join("/srv/app/model/data", model_name)
                try:
                    os.makedirs(modelPath)
                except:
                    print("path already exists")
                model = SentenceTransformer(model_name)
                model.save(modelPath)
                l = model_name
                m = f"Downloaded embedder model {model_name}"
            else:
                print("Model Not supported")
                l = "None"
                m = "Not supported model"
        else:
            print("Not supported")
            l = "None"
            m = "Not supported task"
    else:
        if manager == "pull":
            # Download specified model
            try:
                model_name = param['options']['params']['model_name'].strip('\"')
                uri = f"{ollama_url}/api/pull"
                data = {
                    "name": model_name
                }
                data = json.dumps(data)
                requests.post(uri, data=data)
                m = f'Pulled model {model_name}.'
            except:
                m = f'ERROR during model download.'
            
        elif manager == "delete":
            # Delete specified model
            model_name = param['options']['params']['model_name'].strip('\"')
            uri = f"{ollama_url}/api/delete"
            data = {
                "name": model_name
            }
            data = json.dumps(data)
            requests.delete(uri, data=data)
            m = f'Deleted model {model_name}.'
        else:
            m = "No task specified"
        
        # List all existing models    
        uri = f"{ollama_url}/api/tags"
        response = requests.get(uri).json()
        response = response['models']
        try:
            l = ""
            for r in response:
                l += r['model'].split(":")[0]
                l += " "
        except:
            l = None
    l = [l]
    m = [m]
    cols={'Models': l, 'Message': m}
    returns=pd.DataFrame(data=cols)
    return returns







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

def compute(model,df,param):
    manager = param['params']['task'].strip('\"')
    try:
        model_type = param['params']['model_type'].strip('\"')
    except:
        model_type = "LLM"
    
    if model_type == "embedder_model":
        # embedder model
        if manager == "pull":
            model_name = param['params']['model_name'].strip('\"').strip('\'')
            modelPath = os.path.join("/srv/app/model/data", model_name)
            try:
                os.makedirs(modelPath)
            except:
                print("path already exists")
            model = SentenceTransformer(model_name)
            model.save(modelPath)
            l = model_name
            m = f"Downloaded embedder model {model_name}"
        else:
            print("Not supported")
            l = "None"
            m = "Not supported task"
                  
    else:
        # LLM model
        if manager == "pull":
            # Download specified model
            try:
                model_name = param['params']['model_name'].strip('\"')
                uri = f"{ollama_url}/api/pull"
                data = {
                    "name": model_name
                }
                data = json.dumps(data)
                requests.post(uri, data=data)
                m = f'Pulled model {model_name}.'
            except:
                m = f'ERROR during model download.'
            
        elif manager == "delete":
            # Delete specified model
            model_name = param['params']['model_name'].strip('\"')
            uri = f"{ollama_url}/api/delete"
            data = {
                "name": model_name
            }
            data = json.dumps(data)
            requests.delete(uri, data=data)
            m = f'Deleted model {model_name}.'
        else:
            m = "No task specified"
        
        # List all existing models    
        uri = f"{ollama_url}/api/tags"
        response = requests.get(uri).json()
        response = response['models']
        try:
            l = ""
            for r in response:
                l += r['model'].split(":")[0]
                l += " "
        except:
            l = None

    cols={'Models': l, 'Message': m}
    returns= [cols]
    return returns






