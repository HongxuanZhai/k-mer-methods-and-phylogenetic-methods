from Bio import SeqIO
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from itertools import product
import pyvolve

real_tree = pyvolve.read_tree(file='../data/real_tree.xml')
my_model = pyvolve.Model('nucleotide')
my_partition = pyvolve.Partition(models=my_model, root_sequence="GATAGAA")
my_evolver = pyvolve.Evolver(partitions=my_partition, tree=real_tree)


def sim_real_tree(nsim, ntip, k):
    letters = ['A', 'T', 'C', 'G']
    mers = [''.join(letter) for letter in product(letters, repeat=k)]

    def count_mers(sequence: str, mer: str):  # count the frequency of mer in a sequence
        i = 0
        cnt = 0
        while i + len(mer) - 1 < len(sequence):
            if sequence[i:(i + len(mer))] == mer:
                cnt += 1
            i += 1
        return cnt

    M_0 = coo_matrix((ntip, ntip), dtype=float).toarray()
    for sim in range(nsim):
        i = 0
        my_model = pyvolve.Model('nucleotide')
        my_partition = pyvolve.Partition(models=my_model, root_sequence="GATAGAA")
        my_evolver = pyvolve.Evolver(partitions=my_partition, tree=real_tree)
        my_evolver()
        seq_dict = {rec.id: rec.seq for rec in SeqIO.parse("./simulated_alignment.fasta", "fasta")}
        M = coo_matrix((len(seq_dict), 4 ** k), dtype=float).toarray()
        for key in seq_dict:
            for j in range(len(mers)):
                M[i][j] = count_mers(seq_dict[key], mers[j])
            i += 1
        M_0 += np.matmul(M, M.transpose())
    return M_0 * 1 / nsim


mat1 = sim_real_tree(10000, 50, 1)

mat2 = sim_real_tree(10000, 50, 2)

mat3 = sim_real_tree(10000, 50, 3)

mat4 = sim_real_tree(10000, 50, 4)

mat5 = sim_real_tree(10000, 50, 5)

mat6 = sim_real_tree(10000, 50, 6)

mat7 = sim_real_tree(10000, 50, 7)



def eigen(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    rev_eigvals = eigvals[::-1]
    rev_eigvecs = eigvecs[:, ::-1]
    return rev_eigvals, rev_eigvecs


mat1_val, mat1_vec = eigen(mat1)
mat2_val, mat2_vec = eigen(mat2)
mat3_val, mat3_vec = eigen(mat3)
mat4_val, mat4_vec = eigen(mat4)
mat5_val, mat5_vec = eigen(mat5)
mat6_val, mat6_vec = eigen(mat6)
mat7_val, mat7_vec = eigen(mat7)

df_real_largest = pd.DataFrame({"k1": mat1_vec[:, 0], "k2": mat2_vec[:, 0],
                                "k3": mat3_vec[:, 0], "k4": mat4_vec[:, 0],
                                "k5": mat5_vec[:, 0], "k6": mat6_vec[:, 0],
                                "k7": mat7_vec[:, 0]})
df_real_largest.to_csv("../data/real_largest.csv", index=False)

df_real_second = pd.DataFrame({"k1": mat1_vec[:, 1], "k2": mat2_vec[:, 1],
                                "k3": mat3_vec[:, 1], "k4": mat4_vec[:, 1],
                                "k5": mat5_vec[:, 1], "k6": mat6_vec[:, 1],
                                "k7": mat7_vec[:, 1]})
df_real_second.to_csv("../data/real_second.csv", index=False)

df_real_third = pd.DataFrame({"k1": mat1_vec[:, 2], "k2": mat2_vec[:, 2],
                                "k3": mat3_vec[:, 2], "k4": mat4_vec[:, 2],
                                "k5": mat5_vec[:, 2], "k6": mat6_vec[:, 2],
                                "k7": mat7_vec[:, 2]})
df_real_third.to_csv("../data/real_third.csv", index=False)

df_real_eigvals = pd.DataFrame({"1": mat1_val, "2": mat2_val,
                                "3": mat3_val, "4": mat4_val,
                                "5": mat5_val, "6": mat6_val,
                                "7": mat7_val})
df_real_eigvals.to_csv("../data/r_eigvals.csv", index=False)
