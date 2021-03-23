#!/usr/bin/python3
# coding=utf-8
# Authors: Sharmila Upadhyaya, Bernadeta Griciūtė
# Emails:  {saup00001,begr00001}@stud.uni-saarland.de
# Organization: Universität des Saarlandes
# Copyright 2020 Sharmila Upadhyaya, Bernadeta Griciūtė 
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





import argparse
from collections import defaultdict 

def main():
    
    parser = argparse.ArgumentParser(description='conll format preprocessing')
    parser.add_argument('conll',type=str,
                    help='conll format file', default = "sample/all.txt")
    parser.add_argument('output_file',default = "sample.tsv",type=str,help='name of the output file to be saved')
    parser.add_argument('output_info',default = "sample.info",type=str,help='name of the output file to be save information about data')
    args = parser.parse_args()

    data = open(args.conll, "r").readlines()
    new_data = open(args.output_file, 'w')
    word_tag = []
    new_data.write("id"+"\t" +"word"+"\t"+"tag"+"\n")
    for line in data:
        line = line.split(" ")
        if "#" in line[0]:
            pass
        elif line[0] == "\n":
            new_data.write("\t".join(["*"]*3) + "\n")
        else:
            each_data = [x.strip() for x in line if x.strip() !='']
            new_data.write("\t".join([x.strip() for x in each_data[2:5]])+"\n")
    
    new_data.close()
    e_seq = 0
    t_seq = 0
    max_seq = 0
    t_seq_length = 0
    min_seq = 100
    tag_dictionary = defaultdict()        
    new_data = open(args.output_file, 'r').readlines()

    for i,line in enumerate(new_data):
        if i==0:
            pass
        else:

            line = line.split("\t")
            if line[0] == "*":
                t_seq += 1
                if e_seq > max_seq:
                    max_seq = e_seq
                if e_seq < min_seq:
                    min_seq = e_seq
                e_seq = 0
        
            else:
                if line[2] in tag_dictionary:
                    tag_dictionary[line[2]] += 1
                else:
                    tag_dictionary[line[2]] = 1
                e_seq += 1
                t_seq_length += 1
     
    mean_seq_length = t_seq_length/(t_seq+1)

    info_data = open(args.output_info, 'w')
    info_data.write("Max sequence length: "+ str(max_seq)+"\n")
    info_data.write("Min sequence length: "+ str(min_seq)+"\n")
    info_data.write("Mean sequence length: "+ str(mean_seq_length)+"\n")
    info_data.write("Number of sequences: "+ str(t_seq)+"\n")
    print("total sequences: ",str(t_seq))
    print("max sequence: ", str(max_seq))
    print("mean sequence length: ", str(mean_seq_length))
    print("min sequence: ", str(min_seq))

    info_data.write("Tags:"+"\n")
    total_tags = sum(list(tag_dictionary.values()))
    for k, v in tag_dictionary.items():
        info_data.write(k.strip()+ "\t" + str(round(v/total_tags, 2))+"%\n")

main()


