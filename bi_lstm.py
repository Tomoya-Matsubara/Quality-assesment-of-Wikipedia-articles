# from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tensorflow.keras.preprocessing import text, sequence

# import csv
import numpy as np
# from sklearn import metrics, cross_validation
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# import tensorflow as tf
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.embedding_ops import embedding


# import random
# import time

# import string

import os

data_dir = "text"            # directory contains text documents
model_size = 20000           # length of output vectors
nb_epochs = 1              # number of training epochs
embedding_size = 30
label_file = "enwikilabel"
MAX_FILE_ID = 100
# cell_size = [256, 128, 128]   # number of neurons per each layer
cell_size = [128]
dropout_ratio = 0.5
dynamic = True                # use dynamic LSTM or not
activation_function = "relu"
learning_rate = 0.001
test_ratio = 0.2

batch_size = 32               # mini-batch training



num_words = 20000    # Maximum number of words
oov_token = '<UNK>' # Encoding for unknown words
pad_type = 'post'
trunc_type = 'pre'

maxlen = 2000       # Maximum number of vector dimension


qualities = ["stub", "start", "c", "b", "ga", "fa"]


"""   Load labels from 'enwikilabel'   """
def load_label(label_file):
    with open(label_file) as f:
        return f.read().splitlines()

"""   Load Wikipedia page contents from 'text/...'   """
def load_content(file_name):
    with open(file_name) as f:
        return f.read()



print('Read labels')
Y = load_label(label_file)

# Convert labels (stub, start, etc) into indices (0, 1, etc)
for i in range(len(Y)):
    Y[i] = qualities.index(Y[i])

Y = Y[:2]

print('Read content')

X = []
for i in range(MAX_FILE_ID):                # Explore exhaustively
    file_name = data_dir + '/' + str(i + 1)
    if os.path.isfile(file_name):           # If a corresponding file can be found
        X.append(load_content(file_name))   # save it in Array X

X = X[:2]

print ('Finish reading data')



"""   Dataset division   """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=2021)

# Convert labels (indices) into one-hot encoding
Y_train = np_utils.to_categorical(Y_train, num_classes=len(qualities))  # e.g) stub → 0 → [1, 0, 0, 0, 0, 0]
Y_test = np_utils.to_categorical(Y_test, num_classes=len(qualities))

print(X_train[0])




### Process vocabulary

print('Process vocabulary')

tokenizer = text.Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(X_train, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(X_test, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

n_words = num_words


### Models

print('Build model')

# n_words = 20000
# model_size = 2000
# embedding_size = 300

net = input_data([None, model_size])
net = embedding(net, input_dim=n_words, output_dim=embedding_size)


for i in range(len(cell_size)):
    net = bidirectional_rnn(net, BasicLSTMCell(cell_size[i]), BasicLSTMCell(cell_size[i]))
    net = dropout(net, dropout_ratio)

net = fully_connected(net, len(qualities), activation='softmax')
net = regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

print ('Train model')

model = tflearn.DNN(net, tensorboard_verbose=1, tensorboard_dir = "logdir/bi_lstm")

print ('Predict')
model.fit(X_train, Y_train, validation_set=(X_test, Y_test), show_metric=True, batch_size=batch_size, n_epoch = nb_epochs)