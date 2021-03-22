import numpy as np
import pandas as pd
import csv
import argparse



def get_sentences(tsv_datapath):
    
    df = open(tsv_datapath,"r").readlines()
    words = []
    tags = []
    sentences = []
    data = open("data/final_sample.tsv","w")

    data.write("id\twords\ttags\n")
    sen_id = 0
    for row in df:
        row=row.split("\t")
        if row[0] == "*":
            sen_id += 1
            data.write(str(sen_id)+"\t"+" ".join(words) + "\t" + " ".join(tags)+"\n")
            words = []
            tags = []
        else:
            words.append(row[1].strip())
            tags.append(row[2].strip())

    return "data/final_sample.tsv"



def split_data(datapath):
    """

    """

    data= open(datapath, "r").readlines()
    train, validate, test = \
            np.split(data[1:],
                       [int(.6*len(data)), int(.8*len(data))])
    header = data[0]
    train_path = "data/train.tsv"
    validate_d = "data/validate.tsv"
    test_d = "data/test.tsv"

    train_d= open(train_path, "w")
    validate_d= open(validate_d, "w")
    test_d= open(test_d, "w")

    train_d.write(data[0])
    validate_d.write(data[0])
    test_d.write(data[0])

    for each in train:
        train_d.write(each)

    for each in validate:
        validate_d.write(each)

    for each in test:
        test_d.write(each)

    return train_d, validate_d, test_d


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='getting sentences in each line and splitting into train, test and validation data')
    parser.add_argument('-tsv_datapath',default ="data/sample.tsv",type=str,help='path of the sample tsv created through data preprocess file')
    args = parser.parse_args()

    final_path = get_sentences(args.tsv_datapath)
    train_path, test_path, validation_path =split_data(final_path)
