#!/usr/bin/env python
# coding: utf-8


    
# In[49]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import base64
import io 

import cudf
from cuml.manifold.umap import UMAP as cumlUMAP
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[94]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param











    
# In[97]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    return model







    
# In[99]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "no fit needed"}
    return info







    
# In[101]:


# apply your model
# returns the calculated results
def plot_to_base64(plot):
    pic_IObytes = io.BytesIO()
    if hasattr(plot,'fig'):
        plot.fig.savefig(pic_IObytes, format='png')
    elif hasattr(plot,'figure'):
        plot.figure.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    pic_IObytes.close()
    return pic_hash

def plot_datashader_as_base64(df,param):
    cat = param['target_variables'][0]
    dfr = df.astype({cat: 'category'})
    cvs = ds.Canvas(plot_width=300, plot_height=300)
    agg = cvs.points(dfr, 'UMAP1', 'UMAP2', ds.count_cat(cat))
    color_key_dga = {'dga':'red', 'legit':'blue'}
    #img = tf.shade(agg, cmap='darkblue', how='log') #, cmap=color_key_dga, how="eq_hist")

    img = tf.shade(agg, cmap=color_key_dga, how="eq_hist")

    img.plot()
    pic_IObytes = img.to_bytesio()
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return str(pic_hash)


def plot_scatter_as_base64(df,param):
    hue=None
    if 'options' in param:
        if 'target_variable' in param['options']:
            hue=str(param['options']['target_variable'][0])
    #plot = sns.pairplot(df,hue=hue, palette="husl")
    sns.set()
    plot = sns.scatterplot(x="UMAP1", y="UMAP2", data=df)
    res = str(plot_to_base64(plot))
    return res

def apply(model,df,param):
    # param['options']['model_name']
    dfeatures = df[param['feature_variables']]
    cuml_umap = cumlUMAP()
    #model['umap'] = cuml_umap
    gdf = cudf.DataFrame.from_pandas(df)
    embedding = cuml_umap.fit_transform(gdf)
    result = embedding.rename(columns={0: "UMAP1", 1: "UMAP2"}).to_pandas()
    result_plot = df[param['target_variables']].join(result)
    if 'plot' in param['options']['params']:
        plots = param['options']['params']['plot'].lstrip("\"").rstrip("\"").lower().split(',')
        for plot in plots:
            if plot=='scatter':
                model["plot_scatter"] = plot_scatter_as_base64(result,param)
            elif plot=='datashader':
                model["plot_datashader"] = plot_datashader_as_base64(result_plot,param)
            else:
                continue

    return result_plot









    
# In[104]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model







    
# In[110]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model







    
# In[112]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







