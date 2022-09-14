import pickle
import numpy as np
from time import time
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model

batch_size = 30
nb_classes = 7
nb_epoch = 300

# input image dimensions
img_rows, img_cols = 180, 180
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (6, 6)

# the data, shuffled and split between train and test sets
(training_data1, validation_data1, test_data1) = pickle.load(open('/app/dataset/mug+ck-sem-contempt-180.p','rb'))

(X_train1, y_train1), (X_test, y_test) = (training_data1[0],training_data1[1]),(test_data1[0],test_data1[1])

X_train = np.concatenate((X_train1, validation_data1[0]))
y_train = np.concatenate((y_train1, validation_data1[1]))

#ckecks if backend is theano or tensorflow for dataset format
if K.image_data_format() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


modelo = Sequential()
modelo.add(Conv2D(32, (5,5), input_shape=input_shape, activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
modelo.add(Flatten())
modelo.add(Dense(1024, activation='relu'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.6))
modelo.add(Dense(nb_classes, activation='softmax'))

plot_model(modelo, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
