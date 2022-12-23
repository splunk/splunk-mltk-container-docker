#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands

import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import time
import copy
from tqdm import tqdm
import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
# from torchmetrics.text.rouge import ROUGEScore
# tensorboard related
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import datetime
import logging
import sys
import io
import os
import psutil
import shutil
# Fine-tune parameters initialization
MODEL_NAME = "/srv/app/model/data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_length_src = 400
max_length_target = 200

batch_size_train = 4
batch_size_valid = 4

epochs = 100
patience = 20

MODEL_DIRECTORY = "/"

class T5FineTuner(nn.Module):
    
    def __init__(self, MODEL_NAME):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, local_files_only=True)

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )





    
# In[2]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    print("DEBUG stage call")
    print("DEBUG " + name)
    with open("/srv/notebooks/data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("/srv/notebooks/data/"+name+".json", 'r') as f:
        param = json.load(f) 
    return df, param







    
# In[4]:


def init(df,param):
    tag = "-- process=fine_tuning_progress model={} max_epoch={} -- ".format(param['options']['model_name'], param['options']['params']['max_epochs'])

    print(tag + "Training data loaded with shape: " + str(df.shape))
    print(tag + "Input parameters: ", param['options']['params'])
    print(tag + "Epoch number: " + param['options']['params']['max_epochs'])
    print(tag + "Base model: " + param['options']['params']['base_model'])
    
    print(tag + "Model Initialization: started")
    MODEL_NAME = "/srv/app/model/data/summarization"
    MODEL_NAME = os.path.join(MODEL_NAME, param['options']['params']['lang'], param['options']['params']['base_model'])
    print(tag + "Model file in " + MODEL_NAME)
    model = T5FineTuner(MODEL_NAME)
    model = model.to(device)
    print(tag + "Model Initialization: successfully finished")
    # GPU memory calculation
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load15/os.cpu_count()) * 100
    stat = shutil.disk_usage("/")
    
    print(tag + "#GPU memory --Total memory: {}, --Memory reserved: {}, --Memory allocated: {}. #CPU: {}% occupied. #disk {}".format(t,r,a,cpu_usage,stat))
    
    return model







    
# In[6]:


def fit(model,df,param):  
    tag = "-- process=fine_tuning_progress model={} max_epoch={} -- ".format(param['options']['model_name'], param['options']['params']['max_epochs'])
    if "batch_size" in param['options']['params']:
        print(tag + "setting batch size to ", param['options']['params']['batch_size'])
        batch_size_train = int(param['options']['params']['batch_size'])
        batch_size_valid = int(param['options']['params']['batch_size'])

    def preprocess_text(text):
        text = re.sub(r'[\r\t\n\u3000]', '', text)
        text = text.lower()
        text = text.strip()
        return text

    data = df.query('text.notnull()', engine='python').query('summary.notnull()', engine='python')
    data = data.assign(
        text=lambda x: x.text.map(lambda y: preprocess_text(y)),
        summary=lambda x: x.summary.map(lambda y: preprocess_text(y)))
    # Data conversion
    def convert_batch_data(train_data, valid_data, tokenizer):

        def generate_batch(data):

            batch_src, batch_tgt = [], []
            for src, tgt in data:
                batch_src.append(src)
                batch_tgt.append(tgt)

            batch_src = tokenizer(
                batch_src, max_length=max_length_src, truncation=True, padding="max_length", return_tensors="pt"
            )
            batch_tgt = tokenizer(
                batch_tgt, max_length=max_length_target, truncation=True, padding="max_length", return_tensors="pt"
            )

            return batch_src, batch_tgt

        train_iter = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=generate_batch)
        valid_iter = DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, collate_fn=generate_batch)

        return train_iter, valid_iter
    MODEL_NAME = "/srv/app/model/data/summarization"
    MODEL_NAME = os.path.join(MODEL_NAME, param['options']['params']['lang'], param['options']['params']['base_model'])
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)
    print(tag + "tokenizer intialized")
    print(tag + "Data vectorization: started")

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['summary'], test_size=0.15, random_state=42, shuffle=True
    )

    train_data = [(src, tgt) for src, tgt in zip(X_train, y_train)]
    valid_data = [(src, tgt) for src, tgt in zip(X_test, y_test)]

    train_iter, valid_iter = convert_batch_data(train_data, valid_data, tokenizer)
    print(tag + "Data vectorization: finished.")
    print(tag + "#Training data: " + str(len(train_data)) + ", #Test data: " + str(len(valid_data)))

    # Training function
    def train(model, data, optimizer, PAD_IDX, i):

        model.train()

        loop = 1
        total = len(data)
        losses = 0
        for src, tgt in data:
            optimizer.zero_grad()

            labels = tgt['input_ids'].to(device)
            labels[labels[:, :] == PAD_IDX] = -100

            outputs = model(
                input_ids=src['input_ids'].to(device),
                attention_mask=src['attention_mask'].to(device),
                decoder_attention_mask=tgt['attention_mask'].to(device),
                labels=labels
            )
            loss = outputs['loss']

            loss.backward()
            optimizer.step()
            losses += loss.item()

            print(tag + "Processed {}% of the {}-th epoch. Finished {} out of {} batches. Loss: {} ".format(round(loop/total*100), i, loop, total, round(losses / loop,2)), flush=True)
            loop += 1

        return losses / len(data)

    # Loss function
    def evaluate(model, data, PAD_IDX):

        model.eval()
        losses = 0
        with torch.no_grad():
            for src, tgt in data:

                labels = tgt['input_ids'].to(device)
                labels[labels[:, :] == PAD_IDX] = -100

                outputs = model(
                    input_ids=src['input_ids'].to(device),
                    attention_mask=src['attention_mask'].to(device),
                    decoder_attention_mask=tgt['attention_mask'].to(device),
                    labels=labels
                )
                loss = outputs['loss']
                losses += loss.item()

        return losses / len(data)

    epochs = int(param['options']['params']['max_epochs'])
    MODEL_DIRECTORY = "/srv/app/model/data/summarization"
    MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, param['options']['params']['lang'], param['options']['model_name'])

    optimizer = optim.Adam(model.parameters())

    PAD_IDX = tokenizer.pad_token_id
    best_loss = float('Inf')
    best_model = None
    counter = 1

    print(tag + 'Model fine-tuning started with {} epochs'.format(epochs))

    for loop in range(1, epochs + 1):

        start_time = time.time()

        loss_train = train(model=model, data=train_iter, optimizer=optimizer, PAD_IDX=PAD_IDX, i=loop)

        elapsed_time = time.time() - start_time

        loss_valid = evaluate(model=model, data=valid_iter, PAD_IDX=PAD_IDX)
        
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load15/os.cpu_count()) * 100
        stat = shutil.disk_usage("/")
        print(tag + "#GPU memory --Total memory: {}, --Memory reserved: {}, --Memory allocated: {}. #CPU: {}% occupied. #disk {}".format(t,r,a,cpu_usage,stat), flush=True)

        print(tag + '[{}/{}] train loss: {:.4f}, valid loss: {:.4f} [{}{:.0f}s] counter: {} {}'.format(
            loop, epochs, loss_train, loss_valid,
            str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
            elapsed_time % 60,
            counter,
            '**' if best_loss > loss_valid else ''
        ),flush=True)

        if best_loss > loss_valid:
            best_loss = loss_valid
            best_model = copy.deepcopy(model)
            counter = 1
        else:
            if counter > patience:
                break

            counter += 1

        tokenizer.save_pretrained(MODEL_DIRECTORY)
        print(tag + "tokenizer saved in " + MODEL_DIRECTORY, flush=True)
        best_model.model.save_pretrained(MODEL_DIRECTORY)
        print(tag + "model saved in " + MODEL_DIRECTORY, flush=True)

    print(tag + "Model fine-tuning successfully finished")
    returns = {}
    return returns







    
# In[4]:


def apply(model,df,param):
    device = torch.device('cpu')
    print("device for apply is {}".format(device))
    if 'beam_size' in param['options']['params']:
        beam_size = int(param['options']['params']['beam_size'])
    else:
        beam_size = 1
    tag = "-- process=apply_progress model={} beam_size={} -- ".format(param['options']['model_name'], beam_size)
    MODEL_DIRECTORY = "/srv/app/model/data/summarization"
    MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, param['options']['params']['lang'], param['options']['model_name'])
    model = {}
    model["tokenizer"] = T5Tokenizer.from_pretrained(MODEL_DIRECTORY)
    model["summarizer"] = T5ForConditionalGeneration.from_pretrained(MODEL_DIRECTORY)
    X = df[param['feature_variables'][0]].values.tolist()
    temp_data=list()
    print(tag + "apply function read inputs")
    for i in range(len(X)):
        batch = model["tokenizer"](str(X[i]), max_length=400, truncation=True, return_tensors="pt")
        outputs = model["summarizer"].generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=400,repetition_penalty=8.0,num_beams=beam_size)
        summary = [model["tokenizer"].decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in outputs]
        temp_data += summary
        print(tag + "finished applying {} out of {} utterances".format(i+1, len(X)))
    cols={"summary": temp_data}
    returns=pd.DataFrame(data=cols)
    print(tag + "apply function successfully finished")
        
    return returns







    
# In[14]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model, name):
    return {}





    
# In[15]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(path):
    model = {}
    return model





    
# In[16]:


# return model summary
def summary(model=None):
    returns = {}
    return returns









