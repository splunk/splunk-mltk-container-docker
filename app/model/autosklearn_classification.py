#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# mltkc_import
# this definition exposes all python module imports that should be available in all subsequent commands
import json
import pandas as pd
import pickle

import autosklearn
#from autosklearn.classification import AutoSklearnClassifier
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

from copy import deepcopy
import re

# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[ ]:


# mltkc_stage
# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[ ]:


# mltkc_init
# initialize the model
# params: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    params = deepcopy(param['options']['params'])
    params.pop('algo', None)
    params.pop('mode', None)
    params.pop('dataset_name', None)
    for key in params:
        try:
            if params[key].isdigit():
                params[key] = int(params[key])
        except:
            pass
    model = {}
    model["model"] = AutoSklearn2Classifier(
        **params
    )
    return model







    
# In[ ]:


# mltkc_stage_create_model_fit
# returns a fit info json object
def fit(model,df,param):
    returns = {}
    for col in df.select_dtypes(['object']):
        df[col] = df[col].astype('category')
    X = df[param['feature_variables']]
    y = df[param['target_variables']].values
    dsname = param['options']['params']['dataset_name'] if ("dataset_name" in param['options']['params']) else None
    returns['dataset_name'] = dsname
    returns['fit_history'] = model["model"].fit(X, y, dataset_name=dsname)
    return returns







    
# In[ ]:


# mltkc_stage_create_model_apply
def apply(model,df,param):
    for col in df.select_dtypes(['object']):
        df[col] = df[col].astype('category')
    X = df[param['feature_variables']]
    y_hat = model["model"].predict(X)
    return y_hat







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    model["summary"] = {}
    model["summary"]["statistics"] = {}
    for s in model["model"].sprint_statistics().split("\n")[1:-1]:
        match = re.search('(.*):\s(.*)', s.strip(), re.IGNORECASE)
        if match:
            model["summary"]["statistics"][match.group(1)] = str(match.group(2))

    cv_result_keys = {'mean_test_score': 1, 'mean_fit_time': 1, 'status': 0, 'rank_test_scores': 1}
    for k,v in cv_result_keys.items():
        model["summary"][k] = str(model["model"].cv_results_[k].tolist()) if (v) else model["model"].cv_results_[k]

    model["summary"]["models"] = []
    models_ww = model["model"].get_models_with_weights()
    p = re.compile('(?<!\\\\)\'')
    for m in models_ww:
        curr_weight = m[0]
        curr_model = p.sub('\"', re.search('.*\((\{.*\})', str(m[1]), re.IGNORECASE).group(1))
        model_json = json.loads(curr_model)
        model_json["weight"] = curr_weight
        model["summary"]["models"].append(model_json)
    pickle.dump(model, open(MODEL_DIRECTORY + name + ".pickle", 'wb'))
    return model







    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    with open(MODEL_DIRECTORY + name + ".pickle", 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model







    
# In[ ]:


# return model summary
def summary(model=None):
    returns = {"version": {"autosklearn": autosklearn.__version__} }
    return returns







