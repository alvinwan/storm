"""
3d convolutional neural network applied to ShapeNet dataset.

Usage:
    cnn3d.py [options]
    cnn3d.py <train> <test> [options]

Options:
    --gpu-id=<gpu_id>   GPU id [default: 0]
    --epochs=<epochs>   Number of epochs [default: 10]
"""

import docopt
import scipy.io
import tensorflow as tf
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.estimator import regression

from math import ceil

from utils.util import load_data

arguments = docopt.docopt(__doc__, version='3d cnn 1.0')
train_path = arguments['<train>']
test_path = arguments['<test>']
epochs = int(arguments['--epochs'])


# Data loading and preprocessing
(X, Y), (X_test, Y_test) = load_data(train_path, test_path, conv=True)


def one_hot(v):
    return np.eye(2)[v.astype(int)].reshape((-1, 2))

Y = np.ravel(Y.T)
Y_test = np.ravel(Y_test.T)

Y_oh = one_hot(Y)
Y_test_oh = one_hot(Y_test)

# Convolutional network building
with tf.device('/gpu:%s' % arguments['--gpu-id']):
    network = input_data(shape=[None, 30, 30, 30, 1])
    network = conv_3d(network, 16, 5, activation=tf.nn.relu)
    network = max_pool_3d(network, 2)
    network = conv_3d(network, 32, 5, activation=tf.nn.relu)
    network = max_pool_3d(network, 2)
    network = fully_connected(network, 128, activation=tf.nn.relu)
    network = fully_connected(network, 2, activation=tf.nn.relu)
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=1e-7)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y_oh, n_epoch=epochs, shuffle=True, validation_set=(X_test, Y_test_oh),
          show_metric=True, batch_size=128, run_id='cnn')


def predict(model, X, batch_size):
    Yhat = None
    num_batches = int(ceil(X.shape[0] / float(batch_size)))
    for i in range(num_batches):
        x = X[i * batch_size: (i + 1) * batch_size]
        yhat = model.predict(x)
        Yhat = yhat if Yhat is None else np.vstack((Yhat, yhat))
    return Yhat

Yhat = np.argmax(predict(model, X, 128), axis=1)
Yhat_test = np.argmax(predict(model, X_test, 128), axis=1)

train_accuracy = np.sum(Yhat == Y) / len(Yhat)
validation_accuracy = np.sum(Yhat_test == Y_test) / len(Yhat_test)
print('Train: %f \t Validation: %f' % (train_accuracy, validation_accuracy))