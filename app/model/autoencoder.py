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
    components = 3
    activation_fn = 'relu'
    # learning_rate = 0.001
    # epsilon=0.00001 # default 1e-07
    if 'options' in param:
        if 'params' in param['options']:
            if 'components' in param['options']['params']:
                components = int(param['options']['params']['components'])
            if 'activation_func' in param['options']['params']:
                activation_fn = param['options']['params']['activation_func']
    input_shape = int(X.shape[1])
    encoder_layer = keras.layers.Dense(components, input_dim=input_shape, activation=activation_fn, kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None), bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))
    decoder_layer = keras.layers.Dense(input_shape, activation=activation_fn, kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None), bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))
    model = keras.Sequential()
    model.add(encoder_layer)
    model.add(decoder_layer)
    #opt = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model







    
# In[ ]:


# mltkc_stage_create_model_fit
# returns a fit info json object
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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # run the training
    returns['fit_history'] = model.fit(x=X,
                                       y=X, 
                                       verbose=2, 
                                       epochs=model_epochs, 
                                       batch_size=model_batch_size, 
                                       #validation_data=(X, Y),
                                       callbacks=[tensorboard_callback])
    # memorize parameters
    returns['model_epochs'] = model_epochs
    returns['model_batch_size'] = model_batch_size
    returns['model_loss_acc'] = model.evaluate(x = X, y = X)
    return returns







    
# In[ ]:


# mltkc_stage_create_model_apply
def apply(model,df,param):
    X = df[param['feature_variables']]
    reconstruction = model.predict(x = X)
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)
    hidden = intermediate_layer_model.predict(x = X)
    y_hat = pd.concat([pd.DataFrame(reconstruction).add_prefix("reconstruction_"), pd.DataFrame(hidden).add_prefix("hidden_")], axis=1)
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













