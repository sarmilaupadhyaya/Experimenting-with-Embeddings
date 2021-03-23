
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import time
import random
import pickle5 as pickle

BATCH_SIZE = 64

tagsid_path = "data/embeddingnew/labelid.pickle"
PATH = "data/output/model_bilstm64.h5"
file = open(tagsid_path,'rb')
tagstoid = pickle.load(file)
file.close()
print(tagstoid)

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
    for i in range(0, len(train_paths),2):
        train_path = train_paths[i:i+2]
        file = open(train_path[0],'rb')
        dataset1 = pickle.load(file)
        file.close()
        dataset1["labels"] = torch.tensor(dataset1["labels"], dtype=int).narrow(1,0,270)
        if len(train_path) == 2:
            file = open(train_path[1],'rb')
            dataset = pickle.load(file)
            file.close()
            dataset["labels"] = torch.tensor(dataset["labels"], dtype=int).narrow(1,0,270)
            dataset1["embeddings"] = torch.cat((dataset1["embeddings"], dataset["embeddings"]), 0)
            dataset1["labels"] = torch.cat((dataset1["labels"], dataset["labels"]), 0)
        else:
            dataset1["embeddings"] = dataset1["embeddings"].narrow(1,0,270)
        yield dataset1

train_paths = find("train", "data/embeddingnew/")
val_paths = find("validation", "data/embeddingnew/")
test_paths = find("test","data/embeddingnew/")


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, drp):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        lstm_out = self.relu(lstm_out)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


EMBEDDING_DIM = 768
HIDDEN_DIM = 64
OUTPUT_DIM = len(unique_tags)
N_LAYERS = 2
DROPOUT = 0.25
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,OUTPUT_DIM, DROPOUT)

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

optimizer = optim.Adam(model.parameters(), lr = 0.1)
criterion = nn.CrossEntropyLoss(ignore_index = -100)


for i in range(10):
    
    count  = 0
    epoch_loss = 0
    epoch_acc = 0

    train_batches = get_batches(train_paths)
    model.train()
    for each_batch in train_batches:
        x = each_batch["embeddings"]
        y = each_batch["labels"]
        optimizer.zero_grad()
        predictions = model(x).permute(0,2,1)
        loss = criterion(predictions, y)
        acc = categorical_accuracy(predictions, y, -100)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        count += 1

    print("EPOCH "+ str(i) + " Training LOSS: " + str(epoch_loss/count))
    print("EPOCH "+ str(i) + " Training ACCURACY: " + str(epoch_acc/count))
    del train_batches
    import gc
    gc.collect()
    val_batches = get_batches(val_paths)
    val_loss = []
    val_acc = []
    for each_batch in val_batches:

        x = each_batch["embeddings"]
        y = each_batch["labels"]
        predictions = model(x).permute(0,2,1)
        loss = criterion(predictions, y)
        acc = categorical_accuracy(predictions, y, -100)
        val_loss.append(loss.item())
        val_acc.append(acc.item())

    print("EPOCH "+ str(i) + " Validation LOSS: " +  str(sum(val_loss)/len(val_loss)))
    print("EPOCH "+ str(i) + " Validation ACCURACY: " + str(sum(val_acc)/len(val_acc)))
    
test_loss = []
test_acc = []
test_batches = get_batches(test_paths)
for each_batch in test_batches:

    x = each_batch["embeddings"]
    y = each_batch["labels"]
    predictions = model(x).permute(0,2,1)
    loss = criterion(predictions, y)
    acc = categorical_accuracy(predictions, y, -100)
    test_loss.append(loss.item())
    test_acc.append(acc.item())

print("EPOCH "+ str(i) + "Test LOSS: " +  str(sum(test_loss)/len(test_loss)))
print("EPOCH "+ str(i) + "Test ACCURACY: " + str(sum(test_acc)/len(test_acc)))

torch.save(model.state_dict(), PATH)
model  = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,OUTPUT_DIM,DROPOUT)
model.apply(init_weights)
model.load_state_dict(torch.load(PATH))
