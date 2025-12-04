#!/usr/bin/env python
# coding: utf-8


    
# In[3]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm, trange
from typing import List, Tuple
import torch
from datasets import Dataset as HFDataset
from huggingface_hub import snapshot_download
from app.model.patched_decoder_multi_resolution import PatchedTSMultiResolutionDecoder,TimesfmMRConfig
from app.model.timesfm_multi_resolution import TimesFmMRTorch, TimesFmTorch
from timesfm.pytorch_patched_decoder import create_quantiles
from timesfm import TimesFmHparams, TimesFmCheckpoint
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"





    
# In[8]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param











    
# In[25]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):

    hp = TimesFmHparams(
        context_len=512,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        backend="gpu" if torch.cuda.is_available() else "cpu",
    )

    try:
        hf_repo = param['options']['params']['hf_repo'].strip("\"")
    except:
        # Need to change to correct default path
        hf_repo = "cisco-ai/cisco-time-series-model-1.0-preview"

    try:
        local_path = param['options']['params']['local_path'].strip("\"")
    except:
        local_path = None
    print(local_path)
    if local_path:
        try:
            print(local_path)
            ckpt = TimesFmCheckpoint(path=local_path)
            model_inst = TimesFmMRTorch(
                hparams=hp,
                checkpoint=ckpt,
                use_multi_resolution=True,
                use_special_token_s=True,
            )
        except:
            # Load from Huggingface instead
            ckpt = TimesFmCheckpoint(huggingface_repo_id=hf_repo)
            model_inst = TimesFmMRTorch(
                hparams=hp,
                checkpoint=ckpt,
                use_multi_resolution=True,
                use_special_token_s=True,
            )
    else:
        ckpt = TimesFmCheckpoint(huggingface_repo_id=hf_repo)
        model_inst = TimesFmMRTorch(
            hparams=hp,
            checkpoint=ckpt,
            use_multi_resolution=True,
            use_special_token_s=True,
        )

    return model_inst







    
# In[15]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "No model training required"}
    return info







    
# In[21]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    try:
        PREDICTION_LENGTH = int(param['options']['params']['forecast_steps'].strip("\""))
    except:
        PREDICTION_LENGTH = 128 
    try:
        value_field = param['options']['params']['value_field'].strip("\"")
    except:
        cols={'Message': ["ERROR: Please input parameter \'value_field\' indicating the value field of the time series data"]}
        returns=pd.DataFrame(data=cols)
        return returns
    try:
        series_list = [df[value_field].values.tolist()[:-PREDICTION_LENGTH]]
    except:
        cols={'Message': ["ERROR: Failed to load time series data. Make sure your value_field name is correct."]}
        returns=pd.DataFrame(data=cols)
        return returns

    # Aggregation factor for low-resolution (i.e. 1-min -> 60-min)
    agg_factor = 60

    # Inference for forecasting
    mean, full = model.forecast(series_list, agg_factor=agg_factor)

    # Obtain mean and quantiles of the forecasted series
    means = series_list[0] + mean.tolist()[0]
    p10 = series_list[0] + full[0,:,1].tolist() 
    p20 = series_list[0] + full[0,:,2].tolist() 
    p30 = series_list[0] + full[0,:,3].tolist() 
    p40 = series_list[0] + full[0,:,4].tolist() 
    p50 = series_list[0] + full[0,:,5].tolist() 
    p60 = series_list[0] + full[0,:,6].tolist() 
    p70 = series_list[0] + full[0,:,7].tolist() 
    p80 = series_list[0] + full[0,:,8].tolist() 
    p90 = series_list[0] + full[0,:,9].tolist() 

    cols = {'mean': means, 'p10': p10, 'p20': p20, 'p30': p30, 'p40': p40, 'p50': p50, 'p60': p60, 'p70': p70, 'p80': p80, 'p90': p90}

    result = pd.DataFrame(cols)

    return result







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    model = {}
    return model





    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    return model





    
# In[18]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







