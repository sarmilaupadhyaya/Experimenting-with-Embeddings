import argparse
import pandas as pd
import re


data = open("sample/all.txt", "r")

new_data = open("sample/sample.tsv", "w")

def main():
    
    args = parse_argument()
    parser = argparse.ArgumentParser(description='conll format preprocessing')
    parser.add_argument('conll',type=str,
                    help='conll format file', default = "sample.conll")
    parser.add_argument('output_file',type=str,help='name of the output file to be saved',"final.tst")

    args = parser.parse_args()

    data = open(args.conll, "r").readlines()
    new_data = open(args.output_file, 'w')
    for line in data:
        line = line.split(" ")

	    if "#" in line[0]:
		pass
	    elif line[0] == "":
		new_data.write("\t".join(["*"]*3) + "\n")

	    else:
		#each_data = line.split(" ")
		each_data = [x.strip() for x in line if x.strip() !='']
		new_data.write("\t".join([x.strip() for x in each_data[2:5]])+"\n")


	df = pd.read_csv("sample/sample.tsv", sep = "\t", header=None)
	df.columns = ["POSITION","WORD","POS"]
	t_seq = 0
	max_seq = 0
	t_seq_length = 0
	e_seq = 0
	min_seq = 100
        
        new_data = open(args.output_file, 'r')
	for index, row in df.iterrows():

	    if row["POSITION"] == "*":
		t_seq += 1
		if e_seq > max_seq:
		    max_seq = e_seq
		if e_seq < min_seq:
		    min_seq = e_seq
		e_seq = 0
	    else:
		e_seq += 1
		t_seq_length += 1

	mean_seq_length = t_seq_length/t_seq

	print("total sequences: ",str(t_seq))
	print("max sequence: ", str(max_seq))
	print("mean sequence length: ", str(mean_seq_length))
	print("min sequence: ", str(min_seq))
	import pdb
	pdb.set_trace()
	df_count = df["POS"].value_counts()





