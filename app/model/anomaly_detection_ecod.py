#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
#from pyod.models.iforest import IForest
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"













    
# In[ ]:


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
    model = ECOD(contamination=0.01)
    # parallization options for ECOD:
    # ECOD(n_jobs=2)    
    # most of other PyOD models would work similar, e.g. replace with Isolation Forest:
    #model = IForest()

    return model







    
# In[ ]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    X = df[param['feature_variables'][0]]
    X_train = np.reshape(X.to_numpy(), (len(X), 1))

    # contamination = 0.01
    model.fit(X_train)
    
    info = {"message": "model trained"}
    return info







    
# In[ ]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    X = df[param['feature_variables'][0]]
    X_apply = np.reshape(X.to_numpy(), (len(X), 1))
    
    y_hat = model.predict(X_apply)  # outlier labels (0 or 1)
    y_scores = model.decision_function(X_apply)  # outlier scores

    result = pd.DataFrame(y_hat, columns=['outlier'])
    return result







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    if model is not None:
        if isinstance(model,ECOD):
            from joblib import dump, load
            dump(model, MODEL_DIRECTORY + name + '.joblib')
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    from joblib import dump, load
    model = load(model, MODEL_DIRECTORY + name + '.joblib')
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns













