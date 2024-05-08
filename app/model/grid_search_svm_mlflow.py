#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import re
import joblib
from sklearn.model_selection import train_test_split
import mlflow
import sklearn
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[3]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param

















    
# In[10]:


def log_run(gridsearch: GridSearchCV, experiment_name: str, model_name: str, run_index: int, conda_env, tags={}):
    """Logging of scikit learn grid search cross validation results to mlflow tracking server

    Arguments:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
        tags (dict): Dictionary of extra data and tags (usually features)
    """
    
    cv_results = gridsearch.cv_results_
    with mlflow.start_run(run_name=str(run_index)) as run:  

        mlflow.log_param("folds", gridsearch.cv)

        #print("Logging parameters")       
        for grid in gridsearch.param_grid:
            params = list(grid.keys())
            for param in params:
                mlflow.log_param(param, cv_results["param_%s" % param][run_index])

        #print("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][run_index])
            mlflow.log_metric(score_name.replace("mean","std"), cv_results[score_name.replace("mean","std")][run_index])

        #print("Logging model")        
        mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name) #, conda_env=conda_env)

        #print("Logging extra data related to the experiment")
        mlflow.set_tags(tags) 

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()
        
        #print(mlflow.get_artifact_uri())
        #print("runID: %s" % run_id)

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
    
    # Log all metrics of the gridsearch results into mlflow experiment
    experiment_name = "Gridsearch SVM"
    model_name = param['options']['model_name']
    conda_env = {
        'name': 'mlflow-env',
        'channels': ['defaults'],
        'dependencies': [
            'python=3.8.5',
            'scikit-learn>=0.22.2',
        ]
    }
    tags = {}
    mlflow.set_tracking_uri("http://localhost:6000")
    mlflow.set_experiment(experiment_name)
    for i in range(len(grid_search.cv_results_['params'])):
        log_run(grid_search, experiment_name, model_name, i, conda_env, tags)
    
    # return the best estimator
    model = grid_search.best_estimator_
    return model

# initialize final model
# returns the model object which will be used as a reference to call fit, apply and summary subsequently

def init(df,param):
    model=run_grid_search(df,param)
    return model







    
# In[12]:


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







    
# In[14]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    file = MODEL_DIRECTORY + name + ".pkl"
    joblib.dump(model, file) 
    return model







    
# In[16]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    file = MODEL_DIRECTORY + name + ".pkl"
    model = joblib.load(file)
    return model





    
# In[17]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns





