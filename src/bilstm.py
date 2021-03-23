
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import spacy
import numpy as np

import time
import random
import pickle5 as pickle

BATCH_SIZE = 32

tagsid_path = "data/embeddings/labelid.pickle"
file = open(tagsid_path,'rb')
tagstoid = pickle.load(file)
file.close()

unique_tags = tagstoid.keys()
print(tagstoid.keys())


def find(name, path):
    paths = []
    for root, dirs, files in os.walk(path):
        for efile in files:
            if name in efile:
                paths.append(os.path.join(root, efile))
    return paths


def get_batches(train_paths):
    
    datasets = []
    for train_path in train_paths:
        file = open(train_path,'rb')
        dataset = pickle.load(file)
        file.close()
        yield dataset

train_paths = find("train", "data/embeddings/")
val_paths = find("val", "data/embeddings/")


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


EMBEDDING_DIM = 768
HIDDEN_DIM = 256
OUTPUT_DIM = len(unique_tags)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,OUTPUT_DIM)

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
criterion = nn.CrossEntropyLoss()

# We will be using a simple SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = False) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx)
    correct = max_preds[non_pad_elements].eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr = 3e-4)
criterion = nn.CrossEntropyLoss(ignore_index = -100)


for i in range(10):
    count  = 0
    epoch_loss = 0
    epoch_acc = 0

    train_batches = get_batches(train_paths)
    for each_batch in train_batches:
    
        model.train()
        x = each_batch["embeddings"]
        y = each_batch["labels"]

        optimizer.zero_grad()

        predictions = model(x).permute(0,2,1)


        tags = torch.tensor(y, dtype=int)

        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, -100)

        loss.backward()

        optimizer.step()
    
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        count += 1
    print("EPOCH "+ str(i) + " Training LOSS: " + str(epoch_loss/count))
    print("EPOCH "+ str(i) + " TRaining ACCURACY: " + str(epoch_acc/count))
    val_batches = get_batches(val_paths)
    val_loss = []
    val_acc = []
    for each_batch in val_batches:

        x = each_batch["embeddings"]
        y = each_batch["labels"]
        predictions = model(x).permute(0,2,1)
        tags = torch.tensor(y, dtype=int)
        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, -100)
        val_loss.append(loss)
        val_acc.append(acc)

    print("EPOCH "+ str(i) + " Validation LOSS: " +  str(sum(val_loss)/len(val_loss)))
    print("EPOCH "+ str(i) + " Validation ACCURACY: " + str(sum(val_acc)/len(val_acc)))



