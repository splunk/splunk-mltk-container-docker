#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[9]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[11]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    #model = {}
    #model['hyperparameter'] = 42.0
    model = XGBRegressor()
    return model







    
# In[13]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    returns = {}
    X = df[param['feature_variables']]
    y = df[param['target_variables']]
    #train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    #model.fit(train_X, train_y, verbose=False)
    #predictions = model.predict(test_X)
    #returns['Mean_Absolute_Error'] = str(mean_absolute_error(predictions, test_y))
    
    model.fit(X, y, verbose=False)
    
    info = {"message": "model trained"}
    return info







    
# In[17]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    
    X = df[param['feature_variables']]    
    y_hat = model.predict(X)
    result = pd.DataFrame(y_hat, columns=['predicted_value'])
    
    return result







    
# In[20]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    path = MODEL_DIRECTORY + name + ".json"
    model.save_model(path)
    
    return model





    
# In[21]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = XGBRegressor()
    model.load_model(MODEL_DIRECTORY + name + ".json")
    return model





    
# In[22]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns



