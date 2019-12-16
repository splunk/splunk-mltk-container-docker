#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# mltkc_import
# this definition exposes all python module imports that should be available in all subsequent commands
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

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
    X = df[param['feature_variables']]
    print("FIT build model with input shape " + str(X.shape))
    learning_rate = 0.1
    model_name = "default_linear_regressor"
    if 'options' in param:
        if 'model_name' in param['options']:
            model_name = param['options']['model_name']
        if 'params' in param['options']:
            if 'learning_rate' in param['options']['params']:
                learning_rate = int(param['options']['params']['learning_rate'])

    feature_columns = []
    for feature_name in param['feature_variables']:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
    model = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[32, 16, 8],
        model_dir=MODEL_DIRECTORY + model_name + "/",
    )
    return model







    
# In[ ]:


# mltkc_stage_create_model_fit
# returns a fit info json object
def make_input_fn(df, param, n_epochs=None, batch_size=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((df[param['feature_variables']].to_dict(orient='list'), df[param['target_variables']].values))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
        return dataset.repeat(n_epochs).batch(batch_size)
    return input_fn

def fit(model,df,param):
    returns = {}
    X = df[param['feature_variables']]
    model_epochs = 100
    model_batch_size = 32
    if 'options' in param:
        if 'params' in param['options']:
            if 'epochs' in param['options']['params']:
                model_epochs = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
    # connect model training to tensorboard
    log_dir="/srv/notebooks/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # run the training
    input_fn_train = make_input_fn(df,param,model_epochs,model_batch_size)
    model.train(input_fn=input_fn_train, max_steps=model_epochs)
    # memorize parameters
    returns['model_epochs'] = model_epochs
    returns['model_batch_size'] = model_batch_size
    returns['model_loss_acc'] = model.evaluate(input_fn=input_fn_train)
    return returns







    
# In[ ]:


# mltkc_stage_create_model_apply
def apply(model,df,param):
    X = df[param['feature_variables']]
    model_epochs = 1
    model_batch_size = 32
    if 'options' in param:
        if 'params' in param['options']:
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
    output_fn_train = make_input_fn(df,param,model_epochs,model_batch_size)
    y_hat = pd.DataFrame([p['predictions'] for p in list(model.predict(output_fn_train))])
    return y_hat







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    # model.save(MODEL_DIRECTORY + name + ".h5")
    # serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec([input_column]))
    # export_path = model.export_saved_model(MODEL_DIRECTORY + name +"/", serving_input_fn)
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    # model = keras.models.load_model(MODEL_DIRECTORY + name + ".h5")
    return model





    
# In[ ]:


# return model summary
def summary(model=None):
    returns = {"version": {"tensorflow": tf.__version__, "keras": keras.__version__} }
    if model is not None:
        returns["summary"] = "linear regressor"
    return returns







