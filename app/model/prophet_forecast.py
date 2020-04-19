#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from fbprophet import Prophet
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[35]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[37]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    #X = df[param['feature_variables']]
    #Y = df[param['target_variables']]
    model = Prophet()
    return model







    
# In[45]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    fit_range_start = int(param['options']['params']['fit_range_start'].lstrip("\"").rstrip("\""))
    fit_range_end = int(param['options']['params']['fit_range_end'].lstrip("\"").rstrip("\""))
    df_fit = df[fit_range_start:fit_range_end]
    model.fit(df_fit)
    info = {"message": "model trained on range " + str(fit_range_start)+":"+str(fit_range_end) }
    return info







    
# In[47]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    #future = model.make_future_dataframe(periods=365)
    forecast = model.predict(df)
    return forecast







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    model = {}
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







