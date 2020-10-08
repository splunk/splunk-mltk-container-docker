#!/usr/bin/env python
# coding: utf-8


    
# In[31]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import xgboost
import shap
import matplotlib.pyplot as plt
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[33]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[35]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[37]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    X = df[param['feature_variables']]
    y = df[param['target_variables']] 
    learning_rate = 0.01
    if 'learning_rate' in param['options']['params']:
        learning_rate = float(param['options']['params']['learning_rate'].lstrip("\"").rstrip("\""))
    model['xgboost'] = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    # explain the model's prediction using SHAP values
    model['shap_values'] = shap.TreeExplainer(model['xgboost']).shap_values(X)
    return model







    
# In[51]:


# apply your model
# returns the calculated results
def plot_to_base64(plot):
    import base64
    import io 
    pic_IObytes = io.BytesIO()
    plot.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return pic_hash

def apply(model,df,param):
    X = df[param['feature_variables']]
    result = model['xgboost'].predict(xgboost.DMatrix(X))
    if 'plot' in param['options']['params']:
        plots = param['options']['params']['plot'].lstrip("\"").rstrip("\"").lower().split(',')
        if 'shap_values' in model:            
            shap_values = model['shap_values']
            plt.clf()
            for plot in plots:
                print(plot)
                if plot=='violin':
                    shap.summary_plot(shap_values, X, show=False, plot_type="violin")
                elif plot=='layered_violin':
                    shap.summary_plot(shap_values, X, show=False, plot_type="layered_violin", color='coolwarm')
                elif plot=='bar':
                    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
                else:
                    shap.summary_plot(shap_values, X, show=False)
                # export current plot
                plt.gcf().set_size_inches(10,4)
                plt.tight_layout()
                model["plot_shap"] = plot_to_base64(plt)
    return result







    
# In[11]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    # TODO if needed
    #with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
    #    json.dump(model, file)
    return model







    
# In[13]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    # TODO if needed
    # with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
    #    model = json.load(file)
    return model







    
# In[15]:


# return a model summary
def summary(model=None):
    returns = {"version": {"xgboost": xgboost.__version__, "shap": shap.__version__} }
    return returns





