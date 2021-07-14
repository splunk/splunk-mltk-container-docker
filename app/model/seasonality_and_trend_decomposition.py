#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import statsmodels as sm
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[63]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[13]:


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
    return "info"







    
# In[149]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    data=df
    data['_time']=pd.to_datetime(data['_time'])
    data = data.set_index('_time') # Set the index to datetime object.
    data=data.asfreq('H')
    
    res=STL(data).fit()
    results=pd.DataFrame({"seasonality": res.seasonal, "trend": res.trend, "residual": res.resid})
    results.reset_index(level=0, inplace=True)
    return results







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns











