# Implementing Pretrained BERT EMBEDDING layer for Part of Speech tagging using differnt models and their comparison


## Table of Contents

- [Introduction](#Introduction): information about this project
- [Setup](#Setup): How to set up the project
- [Project Structure](#Project-Structure): File structure of project
- - [Data Preprocessing](#Data-Preprocessing): POS extraction, aggregation
- - [Data Concat & Split](#Data-Concat-&-Split)
- - [Tokenization & Embedding](#Tokenization-&-Embedding): embedding creation
- - [Models](#Models): training

## Introduction

This is the project submitted to Neural Network class of winter 2020. It is an implementation of sequence labelling task for english sentences. We preprocess the data, extract the embedding using bert pretrained model and tokenizer then train the model in different architecture like RNN, LSTM, GRU and BILSTM. The main research idea is to compare the performance of model with each other and also analyse the effect of changing hyperparameters in these model.

#Project Structure

```
```


# Setup

- using pip

```
virtualenv <envname>
pip install -r environment.yaml
```

- using conda

```
conda env update --file environment.yaml
```

We haven't used the GPU so this project works on CPU only. But modification can be done to use GPU which shall be implemented in further days in the same REPO.


#Project-Structure
pass


#Steps-Towards-Tagging
Here are four tasks performed in this project. You can run them individually or at once. 


## Data-Preprocessing

- This step concatenate all files into one,
Script is: `cat data/ontonetes-4.0/*.gold_conll > data/all.conll`. Remember the conll file should be inside data/ontonetes-4.0

- Then it extracts the word id, word and respective tag for it.
- Finally, write the analysis of the data in sample.info file inside data

The step 1 was done manually then for step 2 and 3 we run the script as:

```
./run.sh
```

The parameter to be passed can be edited inside the bash script above. Also, you can find information about arguments to be passed as follow:

```
python src/data_preprocess -h

```

```
usage: data_preprocess.py [-h] conll output_file output_info

conll format preprocessing

positional arguments:
  conll        conll format file
  output_file  name of the output file to be saved
  output_info  name of the output file to be save information about data

```

The result can be viewed inside data/sample.info file


#Data-Concat-And-Split

This step is to generate sentences and tags together which is taken by dataloader in next step. Then, this file is saved as final_sample.tsv. Then, train, test and validation split is done manually and saved as train.csv, test,csv and validate.tsv. These files are inside data/ folder and hardcoded to dataloader. The distribution of train, test and validate dataset are: 60, 20 and 20 percent respectively.


```
python3 src/data_split.py -h
```


```
getting sentences in each line and splitting into train, test and validation data

optional arguments:
  -h, --help            show this help message and exit
  -tsv_datapath TSV_DATAPATH
                        path of the sample tsv created through data preprocess file
```



#Tokenization Embedding
The script to load the dataset , tokenize and save it in embedding directory is inside src/embeddings.py. This script takes default arguments because of lack of memory space to perform embedding by loading bert model in different batch size. Arguments can be seen as:


```
python src/embeddings - h

```

```
Usage: embedding.py [-h] [-tokenizer TOKENIZER] [-embedding_dir EMBEDDING_DIR]
                    [-batch_size BATCH_SIZE] [-model_name MODEL_NAME]

getting embedding for each sentences and saving as pickle

optional arguments:
  -h, --help            show this help message and exit
  -tokenizer TOKENIZER  -name of the bert tokenizer
  -embedding_dir EMBEDDING_DIR
                        directory where you want to save the embeddings
  -batch_size BATCH_SIZE
                        batch size in which data should be saved as pickle. Recommended 32
  -model_name MODEL_NAME
                        name of the bert pretrained model
```

The runtime for saving training, test and validation is respectively ~2 hour, 36 minutes and 39 minutes. This metric is when saved in system with 16 GB RAM. The embedding takes 12 GB RAM and 58 GB Disk space.

The default batch size is 32, but one can try more. In our system 64 batch size crashed. This could be avoided by passing single sentence and saving embeddings. 




#Models

Inside the models.py script we have 4 models namely, LSTM, BILSTM, GRU and RNN.

The script to train the model and test is train_test.py

All the argument in this script is set into default value. If you want to run the train or test then , run script:


```
python train_test.py -h

```

```
usage: train_test.py [-h] [-tagsid_path TAGSID_PATH] [-embedding_dir EMBEDDING_DIR] [-model_type MODEL_TYPE] [-run_type RUN_TYPE] [-hidden_dimension HIDDEN_DIMENSION] [-n_layers N_LAYERS]
                     [-dropout DROPOUT] [-epochs EPOCHS] [-optimizer OPTIMIZER] [-lr LR] [-saved_model SAVED_MODEL]

training and testing the model

optional arguments:
  -h, --help            show this help message and exit
  -tagsid_path TAGSID_PATH
                        -path of pickle file for tag id
  -embedding_dir EMBEDDING_DIR
                        directory where you have saved the embeddings
  -model_type MODEL_TYPE
                        its either bilstm, lstm, gru or rnn
  -run_type RUN_TYPE    either train or test
  -hidden_dimension HIDDEN_DIMENSION
                        number of hidden dimension
  -n_layers N_LAYERS    number of layers
  -dropout DROPOUT      number of layers
  -epochs EPOCHS        number of epochs
  -optimizer OPTIMIZER  adam or sdg
  -lr LR                learning rate
  -saved_model SAVED_MODEL
                        model to test
```

The training is not recommended as it would take around 10-15 minutes for different model. However, if model is trained, it will be saved in the data/output directory


