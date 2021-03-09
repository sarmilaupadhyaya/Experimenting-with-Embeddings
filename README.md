# Experimenting With Embeddings #

This project consists of scripts for our  Neural Network project.

- Part 1:
  - Data Preprocessing
  	reading conll format and extracting only pos and word and its position.
  	analysing the data, getting minimum length of sentence, total sentences, mean length of sentence and distribution of data for each tag.

- Part 2:
 - Not done


## Table of Content:
1. data
    sample.conll
2. src 
    data_preprocess.py
3. environment.yml
4. README.md
5. run.sh


## Virtual Environment Creation

- using pip
```
virtualenv <envname> 
pip install -r environment.yaml
```

- using conda
```
conda env update --file environment.yaml
```

# Modules

## DATA PREPROCESSING
use help to see how to run script.

```
python src/data_preprocess -h
```
conll format preprocessing

positional arguments:
  conll        conll format file
  output_file  name of the output file to be saved
  output_info  name of the output file to be save information about data

optional arguments:
  -h, --help   show this help message and exit

```









```

	- Run script as:

```
./run.sh
```
