#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GRU, Reshape
from keras.utils import plot_model

from GenerateSpectrogramData import (
    GenerateSpectrogramData,
)

from keras import backend as K
K.set_image_dim_ordering('tf')

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
    genre_features.load_deserialize_data()
else:
    genre_features.load_preprocess_data()

shaped_train_X = genre_features.train_X.reshape(genre_features.train_X.shape[0], 1, genre_features.train_X.shape[1], genre_features.train_X.shape[2])
shaped_dev_X = genre_features.dev_X.reshape(genre_features.dev_X.shape[0], 1, genre_features.dev_X.shape[1], genre_features.dev_X.shape[2])
shaped_test_X = genre_features.test_X.reshape(genre_features.test_X.shape[0], 1, genre_features.test_X.shape[1], genre_features.test_X.shape[2])

input_shape = (shaped_train_X.shape[1], shaped_train_X.shape[2], shaped_train_X.shape[3])
print("Build RCNN model ...")
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same', input_shape=input_shape))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'))
model.add(Reshape((192, 128)))
model.add(GRU(32, return_sequences=True))
model.add(GRU(32, return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(len(genre_features.genre_list), activation="softmax"))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("Compiling ...")
opt = keras.optimizers.SGD()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 25  # num of training examples per minibatch
num_epochs = 250
model.fit(
    shaped_train_X,
    genre_features.train_Y,
    batch_size=batch_size,
    epochs=num_epochs,
)

print("\nValidating ...")
score, accuracy = model.evaluate(
    shaped_dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1
)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


print("\nTesting ...")
score, accuracy = model.evaluate(
    shaped_test_X, genre_features.test_Y, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Creates a HDF5 file
model_filename = "rcnn_genre_classifier_lstm_25_250_sgd_100w.h5"
print("\n Saving model: " + model_filename)
model.save(model_filename)
# Creates a json file
print("creating .json file....")
model_json = model.to_json()
f = Path("./rcnn_genre_classifier_lstm_25_250_sgd_100w.json")
f.write_text(model_json)
