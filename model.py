import pickle
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization


def model(input_shape):
    nb_classes = 7
    modelo = Sequential()
    modelo.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
    modelo.add(Flatten())
    modelo.add(Dense(1024, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.6))
    modelo.add(Dense(nb_classes, activation='softmax'))
    sgd = SGD(lr=1e-3)
    # modelo.load_weights("./weights.best.hdf5")

    # modelo.load_weights("/app/weights.current.hdf5")

    modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return modelo