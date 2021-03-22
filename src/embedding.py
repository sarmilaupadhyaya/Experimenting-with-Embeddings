import argparse
import pickle
import datetime
import torch
from transformers import AutoTokenizer,BertModel
from datasets import load_dataset
from torch.utils.data import DataLoader

global label_to_id
global tokenizer

def get_label_list(labels):
    """

    """
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)

    label_list = list(unique_labels)
    label_list.sort()
    return label_list





def tokenize_and_align_labels(examples):
    """

    """

    tokenized_inputs = tokenizer(
            examples["words"],
            padding= "max_length",
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

    labels = []
    for i, label in enumerate(examples["tags"]):
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]] )
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_batchsize(dataset, batch_size):
    """

    """
    
    for i in range(0, len(dataset),batch_size):
        x = dataset[i: i+batch_size]
        if len(x) != batch_size:
            yield x



def save_embedding(datatype, model, dataset, embedding_dir, batch_size):
    """

    """
    tokenized_data = dataset[datatype].map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=2,
            load_from_cache_file=True
        )
    databatches = get_batchsize(tokenized_data, batch_size)
    embeddings = []
    count = 0
    for i, dataset in enumerate(databatches):

        copy_dataset = dict()
        copy_dataset["labels"] = dataset["labels"].copy()

        first_input, second_input = dataset["input_ids"][:int(batch_size/2)], dataset["input_ids"]\
                [int(batch_size/2):]
        output = model(
                    torch.LongTensor(first_input),
        output_hidden_states=True
            )
        embeddings = output.hidden_states[0]
        del output
        if len(second_input) >0:
            output = model(
            torch.LongTensor(second_input),
            output_hidden_states=True
            )
            embeddings = torch.cat((embeddings, output.hidden_states[0]), 0)
            del output
        copy_dataset["embeddings"] = embeddings
        del embeddings
        del dataset
        
        with open(embedding_dir +datatype +str(i) +'.pickle', 'wb') as handle:
            pickle.dump(copy_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            del copy_dataset
        count += 1
        print(datatype + " embedding saved: "+  str(count))

     

def main():

    parser = argparse.ArgumentParser(description='getting embedding for each sentences and saving as pickle')
    parser.add_argument('-tokenizer',type=str,
                    help="-name of the bert tokenizer", default = "bert-base-cased")
    parser.add_argument('-embedding_dir',default = "data/embeddings/",type=str,help='directory where you want to save the embeddings')
    parser.add_argument('-batch_size',default = 32,type=str,help='batch size in which data should be saved as pickle. Recommended 32')
    parser.add_argument('-model_name',default ="bert-base-cased",type=str,help='name of the bert pretrained model')
    args = parser.parse_args()
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
	     args.tokenizer,
	    )
	
    print("tokenizer leaded")
    model = BertModel.from_pretrained(args.model_name)
    print("bert model loaded")
    dataset = load_dataset('dataset_loader.py')
    print("dataset loaded")
    
    label_list = get_label_list(dataset["train"]["tags"]+dataset["validation"]["tags"]+dataset["test"]["tags"])
    global label_to_id
    label_to_id = {l: i for i, l in enumerate(label_list)}
    
    with open(args.embedding_dir +'labelid.pickle', 'wb') as handle:
        pickle.dump(label_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    num_labels = len(label_list)
    
    a = datetime.datetime.now()
    save_embedding("train",model, dataset, args.embedding_dir, args.batch_size)
    print("Total time to save training data:" + str((datetime.datetime.now()-a).total_seconds()))
    a = datetime.datetime.now()
    save_embedding("test",model, dataset, args.embedding_dir, args.batch_size)
    print("Total time to save test data:" + str((datetime.datetime.now()-a).total_seconds()))
    a = datetime.datetime.now()
    save_embedding("validation",model, dataset, args.embedding_dir, args.batch_size)
    print("Total time to save validation data:" + str((datetime.datetime.now()-a).total_seconds()))




if __name__=="__main__":

    main()
