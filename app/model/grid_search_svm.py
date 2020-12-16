#!/usr/bin/env python
# coding: utf-8


    
# In[78]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import re
import joblib
from sklearn.model_selection import train_test_split
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[80]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param

















    
# In[87]:


def split_dataframe(df,param):
    # separate target variable and feature variables
    df_labels = np.ravel(df[param['options']['target_variable']])
    df_features = df[param['options']['feature_variables']]
    return df_labels,df_features

def run_grid_search(df, param):
    df_labels,df_features = split_dataframe(df,param)
    #get GridSearch parameters from Splunk search
    my_grid = param['options']['params']['grid']
    my_grid = my_grid.strip('\"')
    res = re.findall(r'\{.*?\}', my_grid)
    array_res = np.array(res)
    param_grid=[]
    for x in res:
        param_grid.append(eval(x))

    #define model
    model = SVR()

    # Perform gridsearch of model with parameters that have been passed to identify the best performing model parameters.
    #
    # Note: a gridsearch can be very compute intensive. The job below has n_jobs set to -1 which utilizes all of the 
    # available cores to process the search in parallel. Remove that parameter to process single-threaded (this will 
    # significantly increase processing time), or change to another value to specify how many processes can run in parallel.

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(df_features, df_labels)
    model = grid_search.best_estimator_
    return model

# initialize final model
# returns the model object which will be used as a reference to call fit, apply and summary subsequently

def init(df,param):
    model=run_grid_search(df,param)
    return model







    
# In[89]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    df_labels,df_features = split_dataframe(df,param)
    model.fit(df_features, df_labels)
    info = {"message": "model trained"}
    return info







    
# In[91]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    X = df[param['feature_variables']]
    y_hat = model.predict(X)
    result = pd.DataFrame(y_hat)
    return result







    
# In[101]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    file = MODEL_DIRECTORY + name + ".pkl"
    joblib.dump(model, file) 
    return model







    
# In[103]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    file = MODEL_DIRECTORY + name + ".pkl"
    model = joblib.load(file)
    return model





    
# In[95]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns





