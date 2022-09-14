import pickle
import numpy as np
from time import time

import model

np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from model import model

batch_size = 32
nb_classes = 7
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 150, 150
# number of convolutional filters to use
nb_filters = 4
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (6, 6)

# the data, shuffled and split between train and test sets
(training_data1, validation_data1, test_data1) = pickle.load(
    open('/home/felipe/Documents/rec-facial/dataset/mug-150-70-30.p', 'rb'))

(X_train, y_train), (X_test, y_test), (X_val, y_val) = (training_data1[0], training_data1[1]), (
    test_data1[0], test_data1[1]), (validation_data1[0], validation_data1[1])

# X_train = np.concatenate((X_train1, validation_data1[0]))
# y_train = np.concatenate((y_train1, validation_data1[1]))

# ckecks if backend is theano or tensorflow for dataset format
if K.image_data_format() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'val samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


modelo = model(input_shape)

tensorboard = TensorBoard(log_dir="logs-novo/{}".format(time()))

filepath = "./weights.best.hdf5"
filepathAtual = "./weights.current.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepathAtual, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

# training

train_gen = DataGenerator(X_train, Y_train, batch_size)
val_gen = DataGenerator(X_val, Y_val, batch_size)

modelo.fit(train_gen, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=val_gen,
           callbacks=[tensorboard, checkpoint, checkpoint2], shuffle=True)

# modelo.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_val, Y_val),
#            callbacks=[tensorboard, checkpoint, checkpoint2], shuffle=True)
score = modelo.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
