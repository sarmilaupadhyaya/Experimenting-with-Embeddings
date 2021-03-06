import argparse


def main():
    
    parser = argparse.ArgumentParser(description='conll format preprocessing')
    parser.add_argument('conll',type=str,
                    help='conll format file', default = "sample/all.txt")
    parser.add_argument('output_file',default = "sample.tsv",type=str,help='name of the output file to be saved')
    args = parser.parse_args()

    data = open(args.conll, "r").readlines()
    new_data = open(args.output_file, 'w')
    word_tag = []

    for line in data:
        line = line.split(" ")
        if "#" in line[0]:
        elif line[0] == "\n":
            new_data.write("\t".join(["*"]*3) + "\n")
        else:
            each_data = [x.strip() for x in line if x.strip() !='']
            new_data.write("\t".join([x.strip() for x in each_data[2:5]])+"\n")
    
    new_data.close()
    t_seq = 1
    max_seq = 0
    t_seq_length = 0
    min_seq = 100
        
    new_data = open(args.output_file, 'r').readlines()
    for line in new_data:
        e_seq = 0
        line = line.split("\t")
        if line[0] == "*":
            t_seq += 1
            if e_seq > max_seq:
                max_seq = e_seq
            if e_seq < min_seq:
                min_seq = e_seq
        
        else:
            e_seq += 1
            t_seq_length += 1
    
    mean_seq_length = t_seq_length/(t_seq+1)
    print("total sequences: ",str(t_seq))
    print("max sequence: ", str(max_seq))
    print("mean sequence length: ", str(mean_seq_length))
    print("min sequence: ", str(min_seq))

main()


