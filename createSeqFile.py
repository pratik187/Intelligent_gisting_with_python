import codecs
import numpy as np
#from pickle import dump
from keras.utils import to_categorical
import re
import string
# ### Removed only the date time stamps from the lines which had the dates and kept the remaining as it is.
# #### there was some lines with haphazard code occuring from docker, namingly starting with "[8mha:////"
# #### After discussion with my sir I came to know those are irrelevant, so I removed that also.
# ### Some files are not opening as a single text file so in such cases codecs can be used to open it.
#

import os
log_folder  = 'LogFileAnalysis/raw_logfiles/'

params = { "seq_len":100,
          'bs': 512,
          'shuffle': True}

process = list()
count = 0

for log_file in os.listdir(log_folder):
    print('Reading ', log_file)
    with open (log_folder+log_file,"r", encoding = 'utf-8') as file:
        lines = file.readlines()
        if 'SUCCESS' in lines[-1]:
          lines = [line for line in lines if '8mha:' not in line]
          lines = ' '.join(lines)
          lines = ''.join(filter(lambda x:x in string.printable, lines))
          lines = re.sub('\s+', ' ', lines).strip()
          process.append(lines)

raw_text = " ".join(process)

'''
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
print("After mapping")
rev_map = {mapping[x]:x for x in mapping}

import pickle
with open('vocab_mapping.pkl','wb') as vocab_map:
  pickle.dump(mapping, vocab_map)
'''
import pickle
with open('vocab_mapping.pkl','rb') as vocab_map:
  mapping = pickle.load(vocab_map)
rev_map = {mapping[x]:x for x in mapping}


'''
raw_text_encoding = [mapping[x] for x in raw_text]
print(len(raw_text_encoding))
print(raw_text_encoding[:100])
print('After RAW TEXT')
# #### Created Sequence of 100 characters at once
'''

def decode_first(encoded_array, mapping):
  string=[]
  for x in encoded_array[0]:
    string.append(mapping[x])
  print('DECODED :'+''.join(string))

# going through contents of each file
length = params['seq_len']
bs = params['bs']
jump = 10
file_num=0
for raw_text in process:
  raw_text_encoding =[mapping[x] for x in raw_text]
  # store the encoded sequences in batches
  for i in range(0,len(raw_text),bs*jump):
    encoded_sequences = list()
    for j in range(i+length, i+length+jump*bs, jump):
      encoded_seq = raw_text_encoding[j-length:j+1]
      encoded_sequences.append(encoded_seq)
    encoded_sequences = np.array(encoded_sequences)
    if encoded_sequences.shape != (512,101):
      break
    decode_first(encoded_sequences,rev_map)
    #np.save('data/data_'+str(file_num)+'.npy', np.array(encoded_sequences))
    np.save('val_data/data_'+str(file_num)+'.npy', np.array(encoded_sequences))
    file_num+=1
    if file_num%1000 ==0:
        print('Finished ',i)
    if file_num>100000:
        break
  if file_num>100000:
    break

#print('Total Sequences: %d' % len(encoded_sequences))
