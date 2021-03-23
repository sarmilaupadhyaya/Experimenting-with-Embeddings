# Implementing Pretrained BERT EMBEDDING layer for Part of Speech tagging using differnt models and their comparision


## Table of Contents

- [INTRODUCTION](#INTRODUCTION): information about this project
- [SETUP](#SETUP): How to set up the project
- [PROJECT STRUCTURE](PROJECT STRUCTURE): File structure of project
- STEPS TOWARDS TAGGING(#STEPS TOWARDS TAGGING):
- - [Data Preprocessing](##Data-Preprocessing): POS extraction, aggregation
- - [Tokenization & Embedding](##Tokenization-&-Embedding): embedding creation
- - [Models](##Models): training

## INTRODUCTION

This is the project submitted to Neural Network class of winter 2020. It is an implementation of sequence labelling task for english sentences. We preprocess the data, extract the embedding using bert pretrained model and tokenizer then train the model in different architecture like RNN, LSTM, GRU and BILSTM. The main research idea is to compare the performance of model with each other and also analyse the effect of changing hyperparameters in these model.


# SETUP

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



#STEPS TOWARDS TAGGING
Here are four tasks performed in this project. You can run them individually or at once. 

## Data Preprocessing

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

The result can be viwed inside data/sample.info file


