import numpy as np
import keras
import pickle
import os
from keras.utils import to_categorical


def categorical_cross_entropy(actual, predicted):
  logits = -actual*np.log(predicted)-(1-actual)*np.log(1-predicted)
  sum_logits=np.sum(logits, axis = 1)
  mean_logits = np.mean(sum_logits)
  return mean_logits
  

with open('vocab_mapping.pkl','rb') as vocab_map:
  mapping = pickle.load(vocab_map)


model = keras.models.load_model('checkpoint-01-0.81.hdf5')
val_dir = 'data/'
loss= 0.0
val_directories = os.listdir(val_dir)

for (i,val_ds) in enumerate(val_directories[:10000]): 
  X = np.load(val_dir+val_ds)
  y = to_categorical(X[:,-1], num_classes=len(mapping))
  X=to_categorical(X[:,:-1], num_classes=len(mapping))
  y_pred = model.predict(X)
  loss +=  categorical_cross_entropy(y, y_pred)
  print(loss/(i+1))
print(loss/len(val_directories[:10000]))
  
