import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split


# Split it into training and validation data
datafile_name = "data/data.tsv"
data = pd.read_csv(datafile_name, delimiter="\t" , error_bad_lines=False)
trainingSet, testSet = train_test_split(data, test_size=0.2)
trainingSet.to_csv("data/traindata.tsv", sep='\t')
testSet.to_csv("data/validationdata.tsv", sep='\t')


