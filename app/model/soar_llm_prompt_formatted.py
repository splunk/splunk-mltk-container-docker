#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import requests

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.structured_llm import StructuredLLM
from app.model.llm_utils import create_llm, create_embedding_model
from typing import List
from pydantic import BaseModel, Field
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from app.model.llm_utils import create_llm, create_embedding_model, create_vector_db
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"













    
# In[4]:


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







    
# In[3]:


def apply(model,df,param):
    try:
        llm_service = param['options']['params']['llm_service'].strip("\"")
        print(f"Using {llm_service} LLM service.")
    except:
        llm_service = "ollama"
        print("Using default Ollama LLM service.")

    if llm_service == "ollama": 
        try:
            model_name = param['options']['params']['model_name'].strip("\"")
        except:
            returns=pd.DataFrame({"decision": [None], "reason": ["ERROR: Please specify model_name input for using Ollama LLMs"]})
            return returns 
        llm, m = create_llm(service='ollama', model=model_name)
    else:
        llm, m = create_llm(service=llm_service)
    
    try:
        prompt = param['options']['params']['prompt'].strip("\"")
        context = param['options']['params']['context'].strip("\"")
    except:
        returns=pd.DataFrame({"decision": [None], "reason": ["ERROR: Please specify prompt and context inputs"]})
        return returns 

    query = f'''{prompt}
    ----------------------
    {context}
    '''
    
    try:
        labels = json.loads(param['options']['params']['labels'])
        descriptions = json.loads(param['options']['params']['descriptions'])
        assert len(labels) == len(descriptions)
    except Exception as e:
        returns=pd.DataFrame({"decision": [None], "reason": [f"ERROR loading labels and descriptions: {e}"]})
        return returns

    choices = []

    for i in range(len(labels)):
        choices.append(ToolMetadata(description=descriptions[i], name=labels[i]))

    selector = LLMSingleSelector.from_defaults(llm=llm)
    try:
        selector_result = selector.select(
            choices, query=query
        )
        decision = choices[selector_result.selections[0].index].name
        reason = selector_result.selections[0].reason
        returns=pd.DataFrame({"decision": [decision], "reason": [reason]})
        return returns
    except Exception as e:
        returns=pd.DataFrame({"decision": [None], "reason": [f"ERROR receiving response from LLM: {e}"]})
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





    
# In[21]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns



