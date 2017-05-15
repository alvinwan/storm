"""
Runs PCA to create lower dimensional representation.

Usage:
    pca.py train <data> [<checkpoint_id>] [options]
    pca.py encode <checkpoint_id> <data> <out> [options]

Options:
    --latent=<dims>     Dimensions in latent representation [default: 64]
"""

import docopt
import scipy.io
import sklearn.decomposition
import numpy as np
import joblib
import time
import os

from utils.util import load_data


THRESHOLD = 1500/27000.
CKPT_FORMAT = './checkpoints/{id}/checkpoint_{id}.pkl'
ID_ = str(time.time())[-6:]


def train(X, latent):
    """
    Run PCA on the given training data.

    :param X: Training data.
    :param latent: Number of latent dimensions.
    :return: model
    """
    model = sklearn.decomposition.PCA(n_components=latent)
    model.fit(X)
    return model


def evaluate(model, X_test):
    """
    Make a rough approximation for evaluation accuracy.

    :param model: model
    :param X_test: test data
    """
    threshold = THRESHOLD * np.prod(X_test.shape[1:])
    reconstructed_data = model.transform(X_test)
    differences = [np.sum(np.abs(x - x_pred)) for x, x_pred in
                   zip(X_test, reconstructed_data)]
    correct = float(len([diff for diff in differences if diff < threshold]))
    accuracy = correct/X_test.shape[0]
    print(' * [INFO] Reconstruction accuracy: %f (avg: %f)' % (
        accuracy, np.mean(differences)))


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    data_path = arguments['<data>']
    checkpoint_id = arguments['<checkpoint_id>']
    out = arguments['<out>']
    latent = int(arguments['--latent'])
    is_training = arguments['train']
    (X, Y) = load_data(data_path)
    checkpoint_path = CKPT_FORMAT.format(id=checkpoint_id or ID_)

    if is_training:
        model = train(X, latent)
        os.makedirs(os.path.dirname(checkpoint_path))
        joblib.dump(model, checkpoint_path)
        print(' * [INFO] Saved model to', checkpoint_path)
    else:
        print(' * [INFO] Using model', checkpoint_path)
        model = joblib.load(checkpoint_path)
        encoded_train = model.transform(X)
        scipy.io.savemat(out, {'X': encoded_train, 'Y': Y})
        print(' * [INFO] Saved featurized data to', out)