"""
Generates the shapes dataset and provides basic utilities.

Usage:
    geometries.py [options]

Options:
    --n=<n>         Number of samples [default: 300]
    --dim=<dim>     dim * dim * dim for each sample. [default: 20]
"""


from collections import Counter

import numpy as np
import os
import scipy.io

from utils.util import diamond
from utils.util import sphere
from utils.util import cube
from utils.util import noise
from utils.util import plot3d

import docopt


SEED = int(hash('machinelearning')) % (2**32 - 1)
OUTPUT_FILE_SUFFIX = '_%dx.mat'
PLOT_FILE_PREFIX = '%dx_'
CONSTRUCTORS = {
    'diamond': diamond,
    'cube': cube,
    'sphere': sphere
}

#####################
# DATASET UTILITIES #
#####################

np.random.seed(SEED)


def generate_dataset(n=300, dim=20, num_plot=3, display_step=20,
                     noise_level=0.01, splits=('train', 'test'),
                     logdir='../data/generated/'):
    data = None
    global_unique_hashes = set()
    for split in splits:
        unique_hashes_in_split = set()
        class_counts = Counter()

        Xs, ys = None, None
        append = lambda M, v: v if M is None else np.vstack((M, v))

        for i in range(n):
            if i % display_step == 0:
                print(split, i)

            label = np.random.choice(['diamond', 'sphere', 'cube'])
            if label == 'diamond':
                size = np.random.choice(range(2, 12))
            elif label == 'sphere':
                size = np.random.choice(range(2, 12))
            elif label == 'cube':
                size = np.random.choice(range(2, 21, 2))
            else:
                assert False

            constructor = CONSTRUCTORS[label]  # Get function for creating shape
            shape = constructor(size, dim=dim)

            if noise_level > 0:
                mask = noise(dim=dim, prob=(1 - noise_level))
                shape = np.where(mask == 0, shape, 1)

            shape.shape = (1, dim ** 3)
            Xs, ys = append(Xs, shape), append(ys, label)
            if i < num_plot:
                file_name = os.path.join(
                    logdir, (PLOT_FILE_PREFIX % dim) + split + str(i))
                plot3d(shape.reshape(dim, dim, dim), save=file_name)

            hashed = hash_shape(shape)
            global_unique_hashes.add(hashed)
            unique_hashes_in_split.add(hashed)
            class_counts[label] += 1

        assert Xs.shape == (n, dim ** 3)
        assert ys.shape == (n, 1)
        file_name = split + (OUTPUT_FILE_SUFFIX % dim)
        data = {'X': Xs, 'Y': ys}
        scipy.io.savemat(file_name, data)
        print('Wrote to', file_name)
        print('Class counts:', sorted(class_counts.items()))
        print('Number of unique shapes in split:', len(unique_hashes_in_split))

    print('Global number of unique shapes:', len(global_unique_hashes))
    return data


def load_or_generate_dataset(path, n=100, dim=12):
    """Loads dataset if exists. Generation option is deprecated.

    :param path: Path to dataset.
    :return: List of tensors.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return scipy.io.loadmat(path)
    else:
        return generate_dataset(n=n, dim=dim)


def hash_shape(shape):
    return hash('|'.join(str(x) for x in shape.ravel()))


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='1.0')
    n = int(arguments['--n'])
    dim = int(arguments['--dim'])
    data = generate_dataset(n, dim)
