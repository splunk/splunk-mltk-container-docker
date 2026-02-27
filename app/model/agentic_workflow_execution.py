#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import importlib
import asyncio
import concurrent.futures

# Run async agent in sync
def run_async_in_sync(async_func):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(async_func())).result()
    else:
        return asyncio.run(async_func())
# ...
# global constants
module_path = "app.model.agentic_workflows"
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







    
# In[12]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[6]:


# apply your model
def apply(model,df,param):
    try:
        query = param['options']['params']['query'].strip('\"')
    except:
        result = pd.DataFrame({'Message': ["ERROR: Please input a parameter \'prompt\'."]})
        return result

    try:
        workflow_name = param['options']['params']['workflow_name'].strip('\"')
    except:
        result = pd.DataFrame({'Message': ["ERROR: Please input a parameter \'workflow_name\'."]})
        return result

    try:
        llm_utils_module = importlib.import_module(module_path)
        dynamic_function = getattr(llm_utils_module, workflow_name)
        print(f"Successfully imported '{workflow_name}' from '{module_path}'.")
        print(f"Type of dynamic_function: {type(dynamic_function)}")

        w = dynamic_function(timeout=6000, verbose=False)
        async def agent_run():
            # Run the agent
            response = await w.run(query=query)
            return response
        output = run_async_in_sync(agent_run)
        result = pd.DataFrame({'Response': [output]})
        return result      

    except ImportError as e:
        error = f"Error importing module '{module_path}': {e}"
        result = pd.DataFrame({'Message': [error]})
        return result
    except AttributeError as e:
        error = f"Error finding attribute '{a}' in module '{module_path}': {e}"
        result = pd.DataFrame({'Message': [error]})
        return result
    except Exception as e:
        error = f"An unexpected error occurred: {e}"
        result = pd.DataFrame({'Message': [error]})
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





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







