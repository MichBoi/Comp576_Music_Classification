#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import keras.optimizers
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense

from GenerateSpectrogramData import (
    GenerateSpectrogramData,
)

# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

genre_features = GenerateSpectrogramData()

if (
    os.path.isfile(genre_features.train_X_preprocessed_data)
    and os.path.isfile(genre_features.train_Y_preprocessed_data)
    and os.path.isfile(genre_features.dev_X_preprocessed_data)
    and os.path.isfile(genre_features.dev_Y_preprocessed_data)
    and os.path.isfile(genre_features.test_X_preprocessed_data)
    and os.path.isfile(genre_features.test_Y_preprocessed_data)
):
    print("Preprocessed files exist, deserializing npy files")
    genre_features.load_deserialize_data()
else:
    print("Preprocessing raw audio files")
    genre_features.load_preprocess_data()

print("Training X shape: " + str(genre_features.train_X.shape))
print("Training Y shape: " + str(genre_features.train_Y.shape))
print("Dev X shape: " + str(genre_features.dev_X.shape))
print("Dev Y shape: " + str(genre_features.dev_Y.shape))
print("Test X shape: " + str(genre_features.test_X.shape))
print("Test Y shape: " + str(genre_features.test_Y.shape))

input_shape = (genre_features.train_X.shape[1], genre_features.train_X.shape[2])
print("Build LSTM RNN model ...")
model = Sequential()

model.add(LSTM(units=96, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=192,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=genre_features.train_Y.shape[1], activation="softmax"))

print("Compiling ...")
opt = keras.optimizers.SGD()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 25  # num of training examples per minibatch
num_epochs = 250
model.fit(
    genre_features.train_X,
    genre_features.train_Y,
    batch_size=batch_size,
    epochs=num_epochs,
)

print("\nValidating ...")
score, accuracy = model.evaluate(
    genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1
)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


print("\nTesting ...")
score, accuracy = model.evaluate(
    genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Creates a HDF5 file
model_filename = "lstm_genre_classifier_lstm_25_250_sgd_80w.h5"
print("\nSaving model: " + model_filename)
model.save(model_filename)
# Creates a json file
print("creating .json file....")
model_json = model.to_json()
f = Path("lstm_genre_classifier_lstm_25_250_sgd_80w.json")
f.write_text(model_json)
