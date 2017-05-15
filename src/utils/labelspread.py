"""
Uses label spread to tag unlabeled data.

Usage:
    labelspread.py <file>
    labelspread.py <files> <files>...
"""

from sklearn.semi_supervised import LabelSpreading
from util import load_data
from docopt import docopt
import numpy as np
import scipy.io

DEFAULT_FILES = ('../data/more_molecules_30.mat', '../data/more_molecules_8avg_30.mat')
DEFAULT_ALL_DESTINATION = '../data/all_molecules_spread_30.mat'

def main():
    arguments = docopt(__doc__)
    if arguments['<file>']:
        filenames = [arguments['<file>']]
    else:
        filenames = arguments['<files>'] or DEFAULT_FILES
    print(' * [INFO] Reading', filenames)

    (unlabeled_X, unlabeled_Y), (X_all, Y_all) = label(filenames)

    scipy.io.savemat('../data/more_molecules_spread_30.mat',
                     {'X': unlabeled_X, 'Y': unlabeled_Y},
                     do_compression=True)

    print(' * [INFO] Wrote %d samples to %s' % (
        X_all.shape[0], DEFAULT_ALL_DESTINATION))
    scipy.io.savemat(DEFAULT_ALL_DESTINATION,
                     {'X': X_all, 'Y': Y_all},
                     do_compression=True)


def label(filenames, train_path='../data/train_molecules_30.mat'):
    """
    Label data with the provided filenames.

    :param filenames: List of filenames containing data to label.
    :return: Newly labeled and conglomerate datasets
    """
    unlabeled = [scipy.io.loadmat(fname) for fname in filenames]
    unlabeled_X = np.vstack([data['X'] for data in unlabeled])
    X, Y = load_data(train_path, shape=(-1, 30, 30, 30))

    num_unlabeled = unlabeled_X.shape[0]
    unlabeled_Y = np.zeros(num_unlabeled) - 1
    unlabeled_Y = unlabeled_Y.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    Y_all = np.vstack((Y, unlabeled_Y))

    X_all = np.vstack((X, unlabeled_X))
    X_all = X_all.reshape((-1, 27000))

    label_prop_model = LabelSpreading()
    label_prop_model.fit(X_all, Y_all)
    Y_all = label_prop_model.transduction_
    unlabeled_Y = Y_all[num_unlabeled:]
    return (unlabeled_X, unlabeled_Y), (X_all, Y_all)


if __name__ == '__main__':
    main()
