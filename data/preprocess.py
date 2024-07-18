"""
This script documents how we took the original names.txt dataset
and split it into train.txt, val.txt, test.txt splits.
TLDR we shuffled the names, took the first 1000 for val, rest for train.

Process:
- navigate to this directory (data/)
- run this script
python preprocess.py
- this creates train.txt, val.txt, test.txt
"""
import random

# read in all the names (32,032 names in total)
names = open("data/names.txt", 'r').readlines()
# get a permutation
random.seed(42) 
ix = list(range(len(names)))
random.shuffle(ix)

# 1000, 1000, rest are test, val, train splits
test_names = [names[i] for i in ix[:1000]]
val_names = [names[i] for i in ix[1000:2000]]
train_names = [names[i] for i in ix[2000:]]

# utility function to write a list of names to a file
def write_names(names, filename):
    with open(filename, 'w') as f:
        for name in names:
            f.write(name)

# save the splits to disk
write_names(test_names, "data/test.txt")
write_names(val_names, "data/val.txt")
write_names(train_names, "data/train.txt")
