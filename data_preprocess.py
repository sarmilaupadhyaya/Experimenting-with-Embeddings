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

    for line in new_data:
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
        info_data.write(k.strip()+ "\t" + str(v/total_tags)+"\n")

main()


