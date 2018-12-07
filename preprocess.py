
## Connect it to TPUS
import datetime
import json
import os
import pprint
import random
import string
import sys
import tensorflow as tf

TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
print('TPU address is', TPU_ADDRESS)

from google.colab import auth
auth.authenticate_user()
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())

  with open('/content/adc.json', 'r') as f:
    auth_info = json.load(f)
  tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
  # Now credentials are set for all future sessions on this TPU.




# downlaod the the data from the main repo
! wget https://competitions.codalab.org/my/datasets/download/69a3e8d0-b836-48b8-8795-36a6865a1c04
! unzip 69a3e8d0-b836-48b8-8795-36a6865a1c04
! rm 69a3e8d0-b836-48b8-8795-36a6865a1c04
! mv data.tsv data/data.tsv 
! mv eval1_unlabelled.tsv data/testdata.tsv



# Split it into training and validation data
import pandas as pd
from sklearn.model_selection import train_test_split

datafile_name = "data/data.tsv"
data = pd.read_csv(datafile_name, delimiter="\t" , error_bad_lines=False)
trainingSet, testSet = train_test_split(data, test_size=0.2)
trainingSet.to_csv("data/traindata.tsv", sep='\t')
testSet.to_csv("data/validationdata.tsv", sep='\t')




# Download pre trained bert model to google bucket storage. 
TASK = 'msaic' 
TASK_DATA_DIR = 'data/' + TASK
print('***** Task data directory: {} *****'.format(TASK_DATA_DIR))
!ls $TASK_DATA_DIR
# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
BERT_MODEL = 'uncased_L-12_H-768_A-12' 
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
!gsutil ls $BERT_PRETRAINED_DIR
BUCKET = 'msiac' 
assert BUCKET, 'Must specify an existing GCS bucket name'
OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(BUCKET, TASK)
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
