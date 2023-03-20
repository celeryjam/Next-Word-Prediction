# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:52:19 2023

@author: Jack Pham
"""

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import TextVectorization, Dense, LSTM, Bidirectional,BatchNormalization,Embedding
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import optimizers


file = open("pg5200.txt", "r", encoding = "utf8")


import re

def clean_str(string):
    string = re.sub(r"\n","", string) #remove new-line character
    string = re.sub(r"[^A-Za-z]", " ", string) #remove numbers and symbols
    string = string.strip().lower()
    return string
lines = []

for i in file:
    lines.append(i)
    
data = ""

for i in lines:
    data = ' '. join(lines)
    
data = clean_str(data)
vocab =[]

for i in data.split():
    if i not in vocab:
        vocab.append(i)
        
data = data.split()

    
VOCAB_SIZE = 3000
MAX_SEQUENCE_LENGTH = 250
def tokenisation_by_keras(list_string,binary_vectorize=False, vocab_size=VOCAB_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH,vocab=vocab):
    
    '''
    Keras TextVectorization application of tokenization
    clean the text and vectorize it
    '''
    #string = Input(shape=(1,))
    #print(string)
    if binary_vectorize:
        vectorize_layer = TextVectorization(output_mode='binary',vocabulary=vocab) #max_tokens=vocab_size,
    else:
        vectorize_layer = TextVectorization(output_mode='int',vocabulary=vocab,max_tokens=vocab_size) #max_tokens=vocab_size, output_sequence_length=max_sequence_length    
      
    #Create a model that uses the vectorize text layer
    model = Sequential()
    # Start by creating an explicit input layer. It needs to have a shape of
    # (1,) (because we need to guarantee that there is exactly one string
    # input per batch), and the dtype needs to be 'string'.
    model.add(Input(shape=(1,), dtype=tf.string))
    # The first layer in our model is the vectorization layer. After this
    # layer, we have a tensor of shape (batch_size, max_len) containing
    # vocab indices.
    model.add(vectorize_layer)
    return model.predict(list_string)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,output_dim=32,input_length=3))
    #model.add(Bidirectional(LSTM(64, input_shape=(3,1))))#,return_sequences=True
    model.add(LSTM(64,input_shape=(3,1),activation='relu'))
    #model.add(LSTM(64))
    #model.add(LSTM(1000))
    #model.add(Dense(1000, activation="relu"))
    model.add(Dense(len(vocab)+2,activation='softmax'))
    opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'])
    return model

data = tokenisation_by_keras(data)

X=[]
y=[]
for i in range(len(data)-3):
    X.append(data[i:i+3])
    y.append(data[i+1])
X = np.array(X)
y = np.array(y)
y = keras.utils.to_categorical(y, num_classes=len(vocab)+2)
model = build_lstm_model()
train_hist = model.fit(X, y, epochs=150, batch_size=16, verbose=1,callbacks=[tensorboard_callback])
