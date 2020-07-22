#!/usr/bin/env python
# coding: utf-8

# ## Data pre processing on a single file


import codecs
import numpy as np
#from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import TimeDistributed, Dropout, Bidirectional, Dense, Embedding, Flatten, Input
from keras.layers import LSTM
from keras.models import Model, Input
 

import numpy as np
import keras

num_data_files = 37684

# Parameters
params = {'dim': (512,100,10), # 512-bs, 100-seq len, 10 -sze of embedding
          'bs': 512,
          'seq_len':100,
          'shuffle': True}

params = { "seq_len":100,
          'bs': 512,
          'shuffle': True}

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, nLabels, batch_size=512, dim=(512,100,10), shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.nLabels = nLabels
        self.dim = dim # how many characters in the sequence
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)
        #return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #print('Inside get item')
        X = np.load('data/data_' + str(index) + '.npy')
        #y = to_categorical(X[:,1:], num_classes=self.nLabels)
        #X=X[:,:-1]
        y = to_categorical(X[:,-1], num_classes=self.nLabels)
        X=to_categorical(X[:,:-1], num_classes=self.nLabels)
        
        '''
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        '''
        return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)

    '''
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
      '''

# Datasets
labels = np.random.permutation(num_data_files)
train_label_limit = int(0.9*len(labels))
train_labels = labels[:train_label_limit]
val_labels = labels[train_label_limit:]

import pickle
with open('vocab_mapping.pkl','rb') as vocab_map:
  mapping = pickle.load(vocab_map)

# Generators
training_generator = DataGenerator(train_labels, len(mapping))
validation_generator = DataGenerator(val_labels, len(mapping))


# define model
'''
model = Sequential()
model.add(Embedding(len(mapping),10, input_length=100)) # len of vocab, length of embedding, max seq len
model.add(LSTM(128, return_sequences=True))
model.add(Flatten())
model.add(Dense(len(mapping), activation='softmax'))
'''

'''
inp = Input(shape=(100,))
model = Embedding(len(mapping), output_dim=10,
                  input_length=100, mask_zero=True)(inp)  # 10-dim embedding
model = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
out = TimeDistributed(Dense(len(mapping), activation="softmax"))(model)  # a dense layer as suggested by neuralNer

model = Model(inp, out)
'''

model = Sequential()
model.add(LSTM(128, input_shape=(100, len(mapping))))
model.add(Dense(len(mapping), activation='softmax'))
print(model.summary())

model = keras.models.load_model('checkpoint-01-0.81.hdf5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = "checkpoint-new-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=1, nb_epoch =10, steps_per_epoch=3000, validation_steps =100, verbose=1, callbacks=[checkpoint])
                    


