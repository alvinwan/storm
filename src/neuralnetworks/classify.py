"""
Classifier for encoded data.

Usage:
    classify.py [options]
    classify.py <train> <test> [options]

Options:
    --latent=<latent>   latent dimensions [default: 64]
    --gpu-id=<gpu_id>   GPU id [default: 0]
    --epochs=<epochs>   Number of epochs [default: 10]
"""

import docopt
import scipy.io
import numpy as np

import tflearn

from utils.util import shapenet_data

arguments = docopt.docopt(__doc__, version='3d cnn 1.0')
latent = int(arguments['--latent'])
train_path = arguments['<train>']
test_path = arguments['<test>']
epochs = int(arguments['--epochs'])


def load_data(train=None, test=None):
    """
    Load data per command line arguments.

    :param train: Data to train on, labels ignored.
    :param train: Train data to encode, must have labels.
    :param test: Test data to encode, must have labels.
    :return: raw, (X, Y), (X_test, Y_test)
    """
    # Data loading and preprocessing
    if train is None:
        print(' * [INFO] Loading shapenet data...')
        (X, Y), (X_test, Y_test) = shapenet_data()
    else:
        print(' * [INFO] Loading %s data...' % train)
        train_data = scipy.io.loadmat(train)
        X, Y = train_data['X'], train_data['Y']
        test_data = scipy.io.loadmat(test)
        X_test, Y_test = test_data['X'], test_data['Y']
    X = X.reshape((-1, latent))
    X_test = X_test.reshape((-1, latent))
    print(' * [INFO] Data loaded.')
    return (X, Y), (X_test, Y_test)


(X, Y), (X_test, Y_test) = load_data(train_path, test_path)


def one_hot(v):
    return np.eye(2)[v].reshape((-1, 2))

Y_oh = one_hot(Y.T)
Y_test_oh = one_hot(Y_test.T)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, latent])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')
network = tflearn.regression(softmax, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=1e-5)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y_oh, n_epoch=epochs, validation_set=(X_test, Y_test_oh),
          show_metric=True, run_id='nn')

Yhat = np.argmax(model.predict(X), axis=1)
Yhat_test = np.argmax(model.predict(X_test), axis=1)

train_accuracy = np.sum(Yhat == Y) / len(Yhat)
validation_accuracy = np.sum(Yhat_test == Y_test) / len(Yhat_test)
print('Validation: %f \t Train: %f' % (train_accuracy, validation_accuracy))