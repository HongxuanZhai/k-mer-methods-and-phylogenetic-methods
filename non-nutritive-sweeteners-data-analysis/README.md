# Tutorial for k-mer analysis

## Introduction

In this tutorial, we show how to use k-mer-based distances to analyze shotgun metagenome
sequencing data. The data collected are from a recent study on the effect of non-nutritive
sweeteners on the gut microbiome [1]. We will focus on analysis of only stool samples under
two treatments: aspartame and control. The other types of analysis can be done in a similar way.

## Usage
The metadata can be found in
```
metadata_and_accessions.csv
```
We use Jellyfish [2] for doing k-mer counting. The usage of the shell script is
```
./bash input_file k sample_treatment
```
The *input_file* has the names of run accessions where the user trigger k-mer
counting procedures. *k* is the the value of k in k-mer counting, and *sample_treatment*
is the sample type and treatment type of all the run accessions in the *input_file*.

Example:
```
./bash stool_aspartame 6 stool_aspartame
```
This does 6-mer counting for run accessions for stool samples with aspartame treatment.
The outputs are .txt file which has the k-mer expressions and the number of occurrences.
First 8 rows of an output file:
```
>73500
AAAAAAA
>58027
AAAAAAC
>84045
AAAAAAG
>86906
AAAAAAT
```
## Euclidean distance between k-mer spectra

We use a .py file to calculate the distances between each samples from the .txt files
generated from Jellyfish. The usage of the of the program is
```
python3 calculate_dist.py sample_name k output_folder *data_folders
```
The *sample_name* has the run accessions' names on which the Euclidean distances are calculated.
The argument *k* describes which k-mer analysis is done in the run accessions in *sample_name*.
The *output_folder* is the directory where the calculated distance matrix is stored, and *data_folders* are
the directories where the k-mer counting results are stored.

For example:
```
python3 calculate_dist.py aspartame_control.txt 2 /N/distance/ /N/mg-data/stool_aspartame/2 /N/mg-data/stool_control/2
```
will calculate Euclidean distances between k-mer spectrum whose run accessions' names are stored in *aspartame_control.txt*.
The calculated distance matrix will be stored at */N/distance/2*. The stored distance matrices are .csv files that can be easily
integrated into *R* for further multidimensional scaling analysis.

## References

[1] Suez, J., Cohen, Y., Valdes-Mas, R., Mor, U., Dori-Bachash, M., Federici, S., Zmora, N., Leshem, A., Heinemann, M., Linevsky, R., et al. (2022). Personalized microbiome-driven effects of non- nutritive sweeteners on human glucose tolerance. Cell.

[2] Marcais, G. and Kingsford, C. (2011). A fast, lock-free approach for efficient parallel counting of occurrences of k-mers. Bioinformatics, 27(6), 764–770.
