import sys
import os
import math
import numpy as np
import pandas as pd


def distance(s1, s2):
    return math.sqrt(s1.subtract(s2, fill_value=0).pow(2).sum())  


def read_in_kmer(file):
    kmers = []
    counts = []
    with open(file, "r") as fi:
      for line in fi:
          if line.startswith(">"):
            counts.append(int(line[1:].rstrip("\n")))
          else:
            kmers.append(line.rstrip("\n"))
    return pd.Series(data=counts, index=kmers)

def main(sample_name, k, output_folder, *args):
    sample_ids = []
    with open(sample_name, "r") as fi:
      for line in fi:
          sample_ids.append(line.rstrip("\n"))
    kmerMatrix = []
    rowNames = []

    for data_folder in args:
        os.chdir(data_folder)
        for file in os.listdir():
          if file.endswith(".txt") and file.split(".")[0] in sample_ids:
              kmer_dict = read_in_kmer(file)
              
              kmerMatrix.append(kmer_dict)
              rowNames.append(file.split(".")[0])
    n = len(kmerMatrix)
    distMatrix = np.zeros((n, n))
    memo = {}
    for i in range(n):
        for j in range(n):
            if i==j:
                distMatrix[i,j] = 0.0
            elif (j, i) in memo:
                distMatrix[i,j] = memo[(j, i)]
            else:
                memo[(i, j)] = distance(kmerMatrix[i], kmerMatrix[j])
                distMatrix[i,j] = memo[(i, j)]
    dist = pd.DataFrame(distMatrix, index=rowNames, columns=rowNames)
    #print(dist)
    dist.to_csv(output_folder+'/'+sample_name.split(".")[0]+"_"+k+".csv")   

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2], sys.argv[3], *sys.argv[4:])



