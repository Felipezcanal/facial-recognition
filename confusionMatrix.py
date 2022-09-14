import pickle
import numpy as np
from time import time

import model

np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import itertools

batch_size = 30
nb_classes = 7
nb_epoch = 10000

# input image dimensions
img_rows, img_cols = 150, 150
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (6, 6)

# the data, shuffled and split between train and test sets
# (training_data1, validation_data1, test_data1) = pickle.load(open('/app/dataset/os_dois/cohn_dataset.p','rb'))
# (training_data1, validation_data1, test_data1) = pickle.load(open('/app/dataset/mug+ck-180.p','rb'))
(training_data1, validation_data1, test_data1) = pickle.load(open('/home/felipe/Documents/rec-facial/dataset/mug-150-70-30.p', 'rb'))

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

modelo = model.model(input_shape)
modelo.load_weights("./weights.best.hdf5")

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

filepath = "weights.best.hdf5"
filepathAtual = "weights.current.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepathAtual, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

# training
# modelo.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test), callbacks=[tensorboard, checkpoint, checkpoint2],  shuffle=True)
score = modelo.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

Y2_test = np.argmax(Y_test, axis=1)  # Convert one-hot to index
predict_x = modelo.predict(X_test)
y_pred = np.argmax(predict_x,axis=1)
# y_pred = modelo.predict_classes(X_test)
print(classification_report(Y2_test, y_pred))

Y_pred3 = modelo.predict(X_test)
y_pred3 = np.argmax(Y_pred3, axis=1)
Y_test3 = np.argmax(Y_test, axis=1)
print('Confusion Matrix')

cm = confusion_matrix(Y_test3, y_pred3)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./confusionMatrix.png')


plot_confusion_matrix(cm=cm,
                      normalize=False,
                      # target_names = ['Neutro', 'Raiva', 'Desprezo', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa'],
                      target_names=['Neutro', 'Raiva', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa'],
                      title="Confusion Matrix")
