"""
Augment data using both rotations and translations.

Usage:
    augment (azimuth | x | y | z | translation) <n> <n>... [options]

Options:
    --train-path=<path>     Path to train matlab file. [default: ../data/train_molecules_30.mat]
    --verbose               Whether or not to print progress
    --out=<path>            Specify the path for output. [default: ../data/train_molecules_augmented_30]
"""

AUG_AZIMUTH = 'azimuth'
AUG_X = 'x'
AUG_Y = 'y'
AUG_Z = 'z'
AUG_TRANSLATION = 'translation'
AUGMENTATIONS = (AUG_AZIMUTH, AUG_X, AUG_Y, AUG_Z, AUG_TRANSLATION)
DEFAULT_DESTINATION = '../data/train_molecules_augmented_30.mat'

from util import load_data
from docopt import docopt
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from scipy.io.matlab.miobase import MatWriteError
import scipy.io
import numpy as np


def main():
    arguments = docopt(__doc__)
    X, Y = load_data(arguments['--train-path'], shape=(-1, 30, 30, 30))
    out = arguments['--out']

    augmented_X, augmented_Y = augment(
        X, Y,
        augmentation=get_augmentation(arguments),
        deltas=arguments['<n>'],
        verbose=arguments['--verbose']
    )

    filename = out or DEFAULT_DESTINATION
    print(' * [INFO] Generated', augmented_X.shape[0], 'new samples from',
          X.shape[0], 'at', filename)

    import pdb
    pdb.set_trace()
    try:
        scipy.io.savemat(
            filename,
            {'X': augmented_X, 'Y': augmented_Y},
            do_compression=True
        )
    except MatWriteError as e:
        augmented_X = augmented_X.reshape((-1, 27000))
        augmented = np.hstack((augmented_X, augmented_Y))
        np.save(filename, augmented)


def get_augmentation(arguments):
    """
    Extract the augmentation specified in the arguments.

    :param arguments: Dictionary of command-line arguments.
    :return: The augmentation in question.
    """
    operation = None
    for augmentation in AUGMENTATIONS:
        if arguments[augmentation]:
            operation = augmentation
    print(' * [INFO] Using augmentation "%s"' % operation)
    assert operation in AUGMENTATIONS, 'Invalid operation.'
    return operation


def augment(
        X, Y,
        augmentation='azimuth',
        deltas=(60, 120, 180, 240, 300),
        verbose=False):
    """
    Augment the provided data. For translation data, we assume that the first
    dimension details the number of samples and that the second, third, and
    fourth dimensions correspond to x, y, and z respectively.

    :param X: Input data
    :param Y: Input labels
    :param augmentation: Augmentation operation to perform
    :param deltas: List of magnitudes of augmentation to perform
    :param verbose: Whether or not to print per-delta information.
    :return:
    """
    augmented_Xs, augmented_Ys = [], []
    for delta in deltas:
        delta = int(delta)
        if augmentation == AUG_AZIMUTH:
            augmented_x = rotate(X, delta, reshape=False, axes=(1, 2))
            augmented_Xs.append(augmented_x)
            augmented_Ys.append(Y)
        if augmentation in (AUG_X, AUG_TRANSLATION):
            augmented_Xs.append(shift(X, (0, delta, 0, 0)))
            augmented_Xs.append(shift(X, (0, -delta, 0, 0)))
            augmented_Ys.extend([Y, Y])
        if augmentation in (AUG_Y, AUG_TRANSLATION):
            augmented_Xs.append(shift(X, (0, 0, delta, 0)))
            augmented_Xs.append(shift(X, (0, 0, -delta, 0)))
            augmented_Ys.extend([Y, Y])
        if augmentation in (AUG_Z, AUG_TRANSLATION):
            augmented_Xs.append(shift(X, (0, 0, 0, delta)))
            augmented_Xs.append(shift(X, (0, 0, 0, -delta)))
            augmented_Ys.extend([Y, Y])
        if verbose:
            print(' * [INFO] Finished %s with delta %d' % (augmentation, delta))
    augmented_X = np.vstack([X] + augmented_Xs).astype(int)
    augmented_Y = np.vstack([Y] + augmented_Ys).astype(int)
    return augmented_X, augmented_Y


if __name__ == '__main__':
    main()
