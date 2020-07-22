import numpy as np
#from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import TimeDistributed, Dropout, Bidirectional, Dense, Embedding, Flatten, Input
from keras.layers import LSTM
from keras.models import Model, Input
 

import numpy as np
import keras

model = keras.models.load_model('checkpoint-new-02-0.77.hdf5')

from keras.preprocessing.sequence import pad_sequences

import pickle
with open('vocab_mapping.pkl','rb') as vocab_map:
  mapping = pickle.load(vocab_map)
rev_map = {mapping[x]:x for x in mapping}
print(mapping)
print(rev_map)


'''   
# generate a sequence of characters with the model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    curr_pos = len(seed_text)
    print(seed_text)
    # generate a fixed number of characters
    encoded = [mapping[char] for char in in_text]
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    pred_str=[]
    pred_num = []
    for _ in range(n_chars):
        # one hot encode
        #encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1, encoded.shape[1])
        # predict character
        y_hat = model.predict(encoded, verbose=0)
        y_hat=np.argmax(y_hat,axis=1)
        #print(y_hat.shape)
        # integer to character map
        out_char = rev_map[y_hat[0,curr_pos]]
        #print(' '+str(y_hat[0,curr_pos]))
        pred_num.append(y_hat[0,curr_pos])
        pred_str.append(out_char)
        #print(out_char.encode('utf-8'))
        #print(type(encoded[0,curr_pos]))
        #print(type(y_hat[0,curr_pos]))
        encoded[0,curr_pos] = y_hat[0,curr_pos]
        #in_text = in_text[:curr_pos]+y_hat[0,curr_pos]+in_text[curr_pos+1]
        curr_pos+=1
    #print(string)
    import string

    #filtered_string = ''.join(filter(lambda x:x in string.printable, pred_str))
    #print(filtered_string)
    #print(u''.join(string).encode('utf-8').strip())
    print(pred_num)
    print(pred_str)
    return in_text
 
# test start 
generate_seq(model, mapping, 100, 
                   '>', 50)

'''

from keras.preprocessing.sequence import pad_sequences
 
# generate a sequence of characters with the model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1, encoded.shape[1], encoded.shape[2])
        # predict character
        y_hat = model.predict_classes(encoded, verbose=0)
        # integer to character map
        out_char = ''
        for char, index in mapping.items():
            if index == y_hat:
                out_char = char
                break
        in_text += char
    return in_text
 
# test start 
print(generate_seq(model, mapping, 100, 
                   'a', 80))