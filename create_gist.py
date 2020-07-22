import numpy as np
import keras
import pickle
import os
import string
import re
from keras.utils import to_categorical

log_folder  = '/raid/data/Deepa/LogFileAnalysis/raw_logfiles/'

def categorical_cross_entropy(actual, predicted):
  logits = -actual*np.log(predicted)-(1-actual)*np.log(1-predicted)
  sum_logits=np.sum(logits, axis = 1)
  mean_logits = np.mean(sum_logits)
  return mean_logits

def read_file(log_file):
  with open (log_file,"r", encoding = 'utf-8') as fle:
      lines = fle.readlines()
      if 'FAILURE' in lines[-1]:
        lines = [line for line in lines if '8mha:' not in line]
        lines = [''.join(filter(lambda x:x in string.printable, line)) for line in lines]
        lines = [re.sub('\s+', ' ', line).strip() for line in lines]
  return lines

def create_gist(log_file,mapping,model):
  lines = read_file(log_file)
  jump = 1
  length = 100
  print('Number of lines ', len(lines))
  f = open("gist.txt","w")
  for raw_text in lines:
    raw_text_encoding =[mapping[x] for x in raw_text]
    # store the encoded sequences in batches
    loss = 0.0
    count = 0
    for i in range(length,len(raw_text)):
      encoded_seq = raw_text_encoding[i-length:i+1]
      X = np.expand_dims(np.array(encoded_seq),0)
      y = to_categorical(X[:,-1], num_classes=len(mapping))
      X=to_categorical(X[:,:-1], num_classes=len(mapping))
      y_pred = model.predict(X)
      loss +=  categorical_cross_entropy(y, y_pred)
      #print(loss)
      count +=1
    if count > 0:
      #print('LOSS FOR: '+ raw_text+' IS ::: ',loss/count)
      if loss/count < 1.19:
        print(raw_text, loss/count)
        f.write(raw_text + "\n")
  f.close()





with open('vocab_mapping.pkl','rb') as vocab_map:
  mapping = pickle.load(vocab_map)

model = keras.models.load_model('checkpoint-new-01-0.82.hdf5')
log_file = '3.txt'
create_gist(log_file,mapping,model)
