import argparse
import os
import time
import random
import wandb
import pickle5 as pickle
import torch.nn as nn
import torch.optim as optim
import torch
from models import BILSTMTagger, LSTMTagger, GRUNet, RNNTagger

EMBEDDING_DIM = 768
global OUTPUT_DIM 

def load_tagid(tagsid_path):

    file = open(tagsid_path,'rb')
    tagstoid = pickle.load(file)
    file.close()

    unique_tags = tagstoid.keys()
    return tagstoid, unique_tags
    
    

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

def save_model(model, model_type, n_layers,hidden_dim,dropout):
    PATH = "data/output/"+model_type+"_"+str(n_layers)+"_"+str(hidden_dim)+"_"+ str(dropout)+".h5"
    torch.save(model.state_dict(), PATH)

    return PATH





def train(epochs, embedding_dir, model, optimizer, criterion,model_type, n_layers,hidden_dim,dropout):

    train_paths = find("train", embedding_dir)
    val_paths = find("validation", embedding_dir)
    print("TRAINING STARTED")
    for i in range(epochs):
        
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
        wandb.log({"train_loss": epoch_loss/count})
        wandb.log({"train_acc": epoch_acc/count})
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

        wandb.log({"val_loss": val_loss/len(val_loss)})
        wandb.log({"val_acc": val_acc/len(val_acc)})

    path = save_model(model, model_type, n_layers,hidden_dim,dropout)

    if path:
        print("MODEL SAVED UNDER PATH:"+ path)
    return model
        
def load_model(saved_model, EMBEDDING_DIM,OUTPUT_DIM):

    model_type, n_layers, HIDDEN_DIM, DROPOUT = saved_model.split("/")[-1].split(".")[0].split("_")
    print(model_type)
    if model_type == "lstm":
        model = LSTMTagger(int(EMBEDDING_DIM), int(HIDDEN_DIM),int(OUTPUT_DIM),float( DROPOUT),int( n_layers))
    elif model_type == "bilstm":
        model = BILSTMTagger(int(EMBEDDING_DIM), int(HIDDEN_DIM),int(OUTPUT_DIM), float(DROPOUT), int(n_layers))
    elif model_type == "gru":
        model = GRUNet(int(EMBEDDING_DIM), int(HIDDEN_DIM),int(OUTPUT_DIM), float(DROPOUT), int(n_layers))
    elif model_type == "rnn":
        model = RNNTagger(int(EMBEDDING_DIM), int(HIDDEN_DIM),int(OUTPUT_DIM), float(DROPOUT), int(n_layers))


    model.load_state_dict(torch.load(saved_model))

    return model
    




def test(embedding_dir, model, criterion):
    test_paths = find("test", embedding_dir)
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

    print( "Test LOSS: " +  str(sum(test_loss)/len(test_loss)))
    print( "Test ACCURACY: " + str(sum(test_acc)/len(test_acc)))
    
    
    
if __name__=="__main__":


    parser = argparse.ArgumentParser(description='training and testing the model')
    parser.add_argument('-tagsid_path',type=str,
                    help="-path of pickle file for tag id", default = "data/embeddingnew/labelid.pickle")
    parser.add_argument('-embedding_dir',default = "data/embeddings2/",type=str,help='directory where you want to save the embeddings')
    parser.add_argument('-model_type',default = "gru",type=str,help='its either bilstm, lstm, gru or rnn')
    parser.add_argument('-run_type',default ="train",type=str,help='either train or test')
    parser.add_argument('-hidden_dimension',default =64,type=int,help='number of hidden dimension')
    parser.add_argument('-n_layers',default =1,type=int,help='number of layers')
    parser.add_argument('-dropout',default =0,type=float,help='number of layers')
    parser.add_argument('-epochs',default =5,type=int,help='number of epochs')
    parser.add_argument('-optimizer',default ="adam",type=str,help='adam or sdg')
    parser.add_argument('-lr',default =0.1,type=int,help='learning rate')
    parser.add_argument('-saved_model',default ="data/output/bilstm_1_64_0.h5",type=str,help='model to test')

    args = parser.parse_args()
    
    tagstoid, unique_tags=load_tagid(args.tagsid_path)
    OUTPUT_DIM = len(unique_tags) 

    if args.run_type == "train":
        if args.model_type == "lstm":
            model = LSTMTagger(EMBEDDING_DIM, args.hidden_dimension,OUTPUT_DIM, args.dropout, args.n_layers)
        elif args.model_type == "bilstm":
            model = BILSTMTagger(EMBEDDING_DIM, args.hidden_dimension,OUTPUT_DIM, args.dropout, args.n_layers)
        elif args.model_type == "gru":
            model = GRUNet(EMBEDDING_DIM, args.hidden_dimension,OUTPUT_DIM, args.dropout, args.n_layers)
        elif args.model_type == "rnn":
            model = RNNTagger(EMBEDDING_DIM, args.hidden_dimension,OUTPUT_DIM, args.dropout, args.n_layers)
        model.apply(init_weights)

        if args.optimizer == "adam":

            optimizer = optim.Adam(model.parameters(), lr = args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr = 0.1)

        criterion = nn.CrossEntropyLoss(ignore_index = -100)
        train(args.epochs, args.embedding_dir, model, optimizer, criterion, args.model_type, args.n_layers, args.hidden_dimension, args.dropout)
    elif args.run_type=="test":

        if args.saved_model == "":

            print("LOADING THE BEST MODEL:")

        model = load_model(args.saved_model,  EMBEDDING_DIM,OUTPUT_DIM)


        criterion = nn.CrossEntropyLoss(ignore_index = -100)
        
        test(args.embedding_dir, model, criterion)
        



