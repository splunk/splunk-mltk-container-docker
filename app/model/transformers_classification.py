#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# this definition exposes all python module imports that should be available in all subsequent commands

import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import time
import random
import copy
from tqdm import tqdm
import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
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

max_length_src = 500
max_length_target = 200

batch_size_train = 4
batch_size_valid = 4

epochs = 100
patience = 20

MODEL_DIRECTORY = "/"

class BertClassifier(nn.Module):
    """
        Bert Model for classification Tasks.
    """
    def __init__(self, MODEL_NAME, D_out, freeze_bert=False):
        super(BertClassifier,self).__init__()
        D_in, H, D_out = 768, 60, D_out
        self.bert = BertModel.from_pretrained(MODEL_NAME, local_files_only=True)
        self.classifier = nn.Sequential(
                            nn.Linear(D_in, H),
                            nn.ReLU(),
                            nn.Linear(H, D_out))
        # Freeze the Bert Model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self,input_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids,
                           attention_mask = attention_mask)
        last_hidden_state_cls = outputs[0][:,0,:]
        logit = self.classifier(last_hidden_state_cls)
        
        return logit





    
# In[12]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    print("DEBUG stage call")
    print("DEBUG" + name)
    with open("/srv/notebooks/data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("/srv/notebooks/data/"+name+".json", 'r') as f:
        param = json.load(f) 
    return df, param







    
# In[14]:


def init(df,param):
    tag = "-- process=fine_tuning_progress model={} max_epoch={} -- ".format(param['options']['model_name'], param['options']['params']['max_epochs'])

    print(tag + "Training data loaded with shape: " + str(df.shape))
    print(tag + "Input parameters: ", param['options']['params'])
    print(tag + "Epoch number: " + param['options']['params']['max_epochs'])
    print(tag + "Base model: " + param['options']['params']['base_model'])
    print(tag + "Model Initialization: started")
    l = len(list(df.columns)) - 1
    MODEL_NAME = "/srv/app/model/data/classification"
    MODEL_NAME = os.path.join(MODEL_NAME, param['options']['params']['lang'], param['options']['params']['base_model'])
    print(tag + "Model file in " + MODEL_NAME)
    model = BertClassifier(MODEL_NAME, l)
    model = model.to(device)
    print(tag + "Model Initialization: successfully finished")
    # GPU memory calculation
    if torch.cuda.is_available(): 
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
    else:
        t, r, a = 0,0,0
    f = r-a  # free inside reserved
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load15/os.cpu_count()) * 100
    stat = shutil.disk_usage("/")
    
    print(tag + "#GPU memory --Total memory: {}, --Memory reserved: {}, --Memory allocated: {}. #CPU: {}% occupied. #disk {}".format(t,r,a,cpu_usage,stat))
    
    return model







    
# In[16]:


def fit(model,df,param):  
    tag = "-- process=fine_tuning_progress model={} max_epoch={} -- ".format(param['options']['model_name'], param['options']['params']['max_epochs'])
    l = len(list(df.columns)) - 1
    df = df.reindex(sorted(df.columns), axis=1)
    MODEL_DIRECTORY = os.path.join("/srv/app/model/data/classification", param['options']['params']['lang'],param['options']['model_name'])
    if "batch_size" in param['options']['params']:
        print(tag + "setting batch size to ", param['options']['params']['batch_size'])
        batch_size_train = int(param['options']['params']['batch_size'])
        batch_size_valid = int(param['options']['params']['batch_size'])
    else:
        batch_size_train = 4
        batch_size_valid = 4
    # Data preparation
    def text_preprocessing(text):
        if param['options']['params']['lang'] == "en":
            text = text.lower()
            text = re.sub(r"what's", "what is ", text)
            text = re.sub(r"won't", "will not ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"can't", "can not ", text)
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"i'm", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r"\'scuse", " excuse ", text)
            text = re.sub(r"\'\n", " ", text)
            text = re.sub(r"-", " ", text)
            text = re.sub(r"\'\xa0", " ", text)
            text = re.sub('\s+', ' ', text)
            text = ''.join(c for c in text if not c.isnumeric())
            text = re.sub(r'(@.*?)[\s]', ' ', text)
            text = re.sub(r'&amp;', '&', text)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = re.sub(r'[\r\t\n\u3000]', '', text)
            text = text.lower()
            text = text.strip()
        return text
    
    MODEL_NAME = "/srv/app/model/data/classification"
    MODEL_NAME = os.path.join(MODEL_NAME, param['options']['params']['lang'], param['options']['params']['base_model'])
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME,do_lower_case=True)
    print(tag + "tokenizer intialized")

    def preprocessing_for_bert(data):
        input_ids = []
        attention_masks = []   
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),   #preprocess sentence
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= max_length_src  ,             #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length 
            return_attention_mask= True        #Return attention mask 
            )
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        
        #convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids,attention_masks
    
    labels = list(df.columns)
    labels.remove('text')
    X = df.text.values
    y = df[labels].values
    X_train, X_val, y_train, y_val =train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    print(tag + "Data vectorization: started")
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs,train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size_train)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size_valid)
    print(tag + "Data vectorization: finished.")
    print(tag + "#Training data: " + str(len(train_data)) + ", #Test data: " + str(len(val_data)))

    
    def initialize_model(epochs=4):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """

        # Instantiate Bert Classifier
        bert_classifier = model

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                         lr=5e-5, #Default learning rate
                         eps=1e-8 #Default epsilon value
                         )
        # Total number of training steps
        total_steps = len(train_dataloader) * epochs
        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps=0, # Default value
                                                  num_training_steps=total_steps)
        return bert_classifier, optimizer, scheduler
    
    loss_fn = nn.BCEWithLogitsLoss()

    def set_seed(seed_value=42):
        """Set seed for reproducibility.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        
    # Training function
    def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
        """Train the BertClassifier model.
        """
        # Start training loop
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0
            
            total = len(train_dataloader)

            # Put the model into the training mode
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels.float())
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()
                
                print(tag + "Processed {}% of the {}-th epoch. Finished {} out of {} batches. Loss: {} ".format(round(batch_counts/total*100), epoch_i+1, batch_counts, total, round(batch_loss / batch_counts,2)), flush=True)
                
                if (step % 50000 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch
                  
                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)
            
            
            tokenizer.save_pretrained(MODEL_DIRECTORY)
            print(tag + "tokenizer saved in " + MODEL_DIRECTORY, flush=True)
            torch.save(model.state_dict(),os.path.join(MODEL_DIRECTORY, "pytorch_model.pt"))
            print(tag + "model saved in " + MODEL_DIRECTORY, flush=True)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = evaluate(model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch
                
                print(tag + '[{}/{}] train loss: {:.4f}, valid loss: {:.4f}, valid accuracy: {:.4f} [{}{:.0f}s]'.format(
                        epoch_i+1, epochs, avg_train_loss, val_loss, val_accuracy,
                        str(int(math.floor(time_elapsed / 60))) + 'm' if math.floor(time_elapsed / 60) > 0 else '',
                        time_elapsed % 60
                    ), flush=True)

        
    def evaluate(model, val_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = loss_fn(logits, b_labels.float())
            val_loss.append(loss.item())
            
            accuracy = accuracy_thresh(logits.view(-1,l),b_labels.view(-1,l))
        
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy
    
    
    def accuracy_thresh(y_pred, y_true, thresh:float=0.5, sigmoid:bool=True):
        "Compute accuracy when `y_pred` and `y_true` are the same size."
        if sigmoid: 
            y_pred = y_pred.sigmoid()
        return ((y_pred>thresh)==y_true.byte()).float().mean().item()

    set_seed(42)    # Set seed for reproducibility
    bert_classifier, optimizer, scheduler = initialize_model(epochs=int(param['options']['params']['max_epochs']))
    train(bert_classifier, train_dataloader, val_dataloader, epochs=int(param['options']['params']['max_epochs']), evaluation=True)
    

    print(tag + "Model fine-tuning successfully finished")
    returns = {}
    return returns







    
# In[18]:


def apply(model,df,param):
    device = torch.device('cpu')
    tag = "-- process=apply_progress model={} max_epoch={} -- ".format(param['options']['model_name'], param['options']['params']['max_epochs'])
    df = df.reindex(sorted(df.columns), axis=1)
    predict_labels = list(df.columns)
    predict_labels.remove('text')
    l = len(predict_labels)
    
    MODEL_DIRECTORY = "/srv/app/model/data/classification"
    MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, param['options']['params']['lang'], param['options']['model_name'])
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIRECTORY)
    MODEL_NAME = os.path.join("/srv/app/model/data/classification", param['options']['params']['lang'], param['options']['params']['base_model'])
    MODEL_DIRECTORY = os.path.join("/srv/app/model/data/classification", param['options']['params']['lang'],param['options']['model_name'])
    model = BertClassifier(MODEL_NAME,l)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIRECTORY, "pytorch_model.pt"), map_location=torch.device(device)))
    print(tag + "Fine-tuned model reloaded.")
    model.eval()
    
    def text_preprocessing(text):
        if param['options']['params']['lang'] == "en":
            text = text.lower()
            text = re.sub(r"what's", "what is ", text)
            text = re.sub(r"won't", "will not ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"can't", "can not ", text)
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"i'm", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r"\'scuse", " excuse ", text)
            text = re.sub(r"\'\n", " ", text)
            text = re.sub(r"-", " ", text)
            text = re.sub(r"\'\xa0", " ", text)
            text = re.sub('\s+', ' ', text)
            text = ''.join(c for c in text if not c.isnumeric())
            text = re.sub(r'(@.*?)[\s]', ' ', text)
            text = re.sub(r'&amp;', '&', text)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = re.sub(r'[\r\t\n\u3000]', '', text)
            text = text.lower()
            text = text.strip()
        return text

    def preprocessing_for_bert(data):
        input_ids = []
        attention_masks = []
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),   #preprocess sentence
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= max_length_src  ,      #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length 
            return_attention_mask= True        #Return attention mask 
            )
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        
        #convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids,attention_masks
    
    X = df[param['feature_variables'][0]].values.tolist()
    labels = list(df.columns)
    labels.remove('text')
    y = df[labels].values
    train_inputs, train_masks = preprocessing_for_bert(X)
    train_labels = torch.tensor(y)

    
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs,train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size_train, shuffle=False)
    
    all_logits = []
    for batch in train_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)

    probs = all_logits.sigmoid().cpu().numpy()
    returns = pd.DataFrame(probs,columns=predict_labels)
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









