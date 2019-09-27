#!/usr/bin/env python
# coding: utf-8


    
# In[14]:


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







    
# In[16]:


# mltkc_stage
# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[18]:


# mltkc_init
# initialize the model
# params: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    X = df[param['feature_variables']]
    print("FIT build model with input shape " + str(X.shape))
    input_shape = int(X.shape[1])
    model = keras.Sequential()
    model.add(keras.layers.Dense(input_shape, input_dim=input_shape, activation=tf.nn.relu))
    for l in range(0,1):
        model.add(keras.layers.Dense(int(input_shape*2), activation=tf.nn.relu))
    model.add(keras.layers.Dense(int(input_shape), activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model







    
# In[20]:


# mltkc_stage_create_model_fit
# returns a fit info json object
def fit(model,df,param):
    returns = {}
    X = df[param['feature_variables']]
    Y = df[param['target_variables']]
    model_epochs = 100
    model_batch_size = None
    if 'options' in param:
        if 'params' in param['options']:
            if 'epochs' in param['options']['params']:
                model_epochs = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
    # connect model training to tensorboard
    log_dir="/srv/notebooks/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # run the training
    returns['fit_history'] = model.fit(x=X,
                                       y=Y, 
                                       verbose=2, 
                                       epochs=model_epochs, 
                                       batch_size=model_batch_size, 
                                       #validation_data=(X, Y),
                                       callbacks=[tensorboard_callback])
    # memorize parameters
    returns['model_epochs'] = model_epochs
    returns['model_batch_size'] = model_batch_size
    returns['model_loss_acc'] = model.evaluate(x = X, y = Y)
    return returns







    
# In[ ]:


# mltkc_stage_create_model_apply
def apply(model,df,param):
    X = df[param['feature_variables']]
    y_hat = model.predict(x = X, verbose=1)
    return y_hat







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    # save keras model to hdf5 file
    # https://www.tensorflow.org/beta/tutorials/keras/save_and_restore_models
    model.save(MODEL_DIRECTORY + name + ".h5")
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    model = keras.models.load_model(MODEL_DIRECTORY + name + ".h5")
    return model





    
# In[ ]:


# return model summary
def summary(model=None):
    returns = {"version": {"tensorflow": tf.__version__, "keras": keras.__version__} }
    if model is not None:
        # Save keras model summary to string:
        s = []
        model.summary(print_fn=lambda x: s.append(x+'\n'))
        returns["summary"] = ''.join(s)
    return returns













