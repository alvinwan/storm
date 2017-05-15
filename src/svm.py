"""
Runs SVM on the provided featurizations of data.

The provided train data below should be featurized, and dimensions for both the
train and test data must match.

Usage:
    svm.py classify <test_path> <model> [options]
    svm.py train <train_path> <test_path> [options]

Options:
    --out=<out>     File to write classification results to [default: results.csv]
    --C=<C>         C value for SVM [default: 1e-7]
    --kernel=<k>    The kernel to use for SVM. [default: linear]
    --confusion     Output the confusion matrix.
    --grid          Initiate a grid search across hyperparameters.
    --model=<model> Saved model (.pkl)
"""

import sys
import docopt
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use('macosx')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm
import time
import joblib
import csv

from sklearn.metrics import confusion_matrix, accuracy_score
from utils.shapes_dataset import load_or_generate_dataset
from utils.util import plot_confusion_matrix
from utils.util import load_data


# TODO(Alvin): remove hard-coded class names
CLASS_NAMES = (0, 1)

grid = {
    'rbf': (1e-3, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7),
    'poly': (1e-7, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3),
    'linear': (1e-3, 1e-1, 1, 1e1, 1e2, 1e3, 1e4)
}


def main():
    """Run main training and evaluation script."""
    arguments = docopt.docopt(__doc__, version='Kmeans 1.0')
    out = arguments['--out']

    if arguments['classify']:
        X = load_data(arguments['<test_path>'])
        model = joblib.load(arguments['--model'])
        pred_test = predict(model, X)
        with open(out) as f:
            writer = csv.writer(f)
            for i, c in enumerate(pred_test):
                writer.write([i, c])
        print(' * [INFO] Wrote classification results to', out)
        return

    (X, Y), (X_test, Y_test) = load_data(arguments['<train_path>'], arguments['<test_path>'])
    data_train = {'X': X, 'Y': Y}
    data_test = {'X': X_test, 'Y': Y_test}
    assert X.shape[1] == X_test.shape[1], 'Dim of data incompatible'

    if arguments['--grid']:
        best_run, best_accuracy = (), 0
        for kernel in grid:
            for C in grid[kernel]:
                results = run(data_train, data_test, kernel, C,
                    confusion=arguments['--confusion'])
                if results['validation'] > best_accuracy:
                    best_model = results['model']
                    best_accuracy = results['validation']
                    best_run = {
                        'kernel': kernel,
                        'C': C,
                        'train_accuracy': results['train']
                    }
        print(' * [INFO] Best run:', best_accuracy)
        print(' * [INFO] More information:', best_run)
        joblib.dump(best_model, out)
        print(' * [INFO] Saved model to', out)
    else:
        results = run(
            data_train,
            data_test,
            C=float(arguments['--C']),
            kernel=arguments['--kernel'],
            confusion=arguments['--confusion'])
        joblib.dump(results['model'], out)
        print(' * [INFO] Saved model to', out)


def run(data_train, data_test, kernel='rbf', C=1e-7, confusion=False):
    """Run the training script and yield accuracy, confusion

    :param data_train: Featurized training data, contains X and Y
    :param data_test: Featurized test data, contains X and Y
    :param kernel: kernel to apply during classification
    :param C: Weight of misclassification during classification
    :param confusion: Whether or not to print confusion matrix
    :return: None
    """
    model = train(data_train, C=C, kernel=kernel)
    pred_train = predict(model, data_train)
    pred_test = predict(model, data_test)

    train_accuracy = accuracy_score(data_train['Y'], pred_train)
    test_accuracy = accuracy_score(data_test['Y'], pred_test)
    print('[%s][%.2E] \t Accuracy: \t (Train) %.4f \t (Val) %.4f' % (kernel, C, train_accuracy, test_accuracy))

    if confusion:
        cnf_matrix = confusion_matrix(data_test['Y'], pred_test)

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES,
                              title='Confusion matrix, without normalization')
        plt.savefig('../data/generated/cnf-%s.png' % str(time.time())[-6:])

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES, normalize=True,
                              title='Normalized confusion matrix')
        plt.savefig('../data/generated/nm-cnf-%s.png' % str(time.time())[-6:])
    return {
        'model': model,
        'train': train_accuracy,
        'validation': test_accuracy
    }


def predict(model, data_test):
    """Make predictions on the test set, using the provided model.

    :param model: Model to make predictions with.
    :param data_test: Test set to evaluate on.
    :return: All predictions
    """
    predictions = np.ravel(model.predict(data_test['X']))
    data_test['Y'] = np.ravel(data_test['Y'])
    assert data_test['Y'].shape == predictions.shape, \
        (data_test['Y'].shape, predictions.shape)
    return predictions


def train(data, C=1.0, kernel='rbf'):
    """Train the SVM on featurized data.

    :param C: Quantity penalizing misclassifications
    :param data: Featurized data.
    :return: Trained model.
    """
    if kernel == 'linear':
        clf = sklearn.svm.LinearSVC(C=C)
    else:
        clf = sklearn.svm.SVC(C=C, kernel=kernel)
    data['Y'] = np.ravel(data['Y'])
    clf.fit(data['X'], data['Y'])
    return clf


if __name__ == '__main__':
    main()
