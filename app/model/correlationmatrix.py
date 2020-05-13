#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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







    
# In[6]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[8]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "no fit needed"}
    return info







    
# In[10]:


# apply your model
# returns the calculated results
def plot_to_base64(plot):
    import base64
    import io 
    pic_IObytes = io.BytesIO()
    if hasattr(plot,'fig'):
        plot.fig.savefig(pic_IObytes, format='png')
    elif hasattr(plot,'figure'):
        plot.figure.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return pic_hash


def plot_pairplot_as_base64(df,param):
    hue=None
    if 'options' in param:
        if 'target_variable' in param['options']:
            hue=str(param['options']['target_variable'][0])
    plot = sns.pairplot(df,hue=hue, palette="husl")
    return str(plot_to_base64(plot))


def plot_correlationmatrix_as_base64(corr):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15))
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    #plot = sns.heatmap(corr, mask=mask, cmap="Spectral", vmax=1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plot = sns.heatmap(corr, cmap="Spectral", vmax=1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    #plot.figure.savefig("plot.png", format='png')
    return str(plot_to_base64(plot))

def apply(model,df,param):
    # param['options']['model_name']    
    dfeatures = df[param['feature_variables']]
    result = dfeatures.corr() #.reset_index()
    if 'plot' in param['options']['params']:
        plots = param['options']['params']['plot'].lstrip("\"").rstrip("\"").lower().split(',')
        for plot in plots:
            if plot=='matrix':
                model["plot_matrix"] = plot_correlationmatrix_as_base64(result)
            elif plot=='pairplot':
                model["plot_pairplot"] = plot_pairplot_as_base64(df,param)
            else:
                continue

    return result







    
# In[12]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model







    
# In[14]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model







    
# In[19]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns





