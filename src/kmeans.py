"""
Run k-means clustering on voxelized data.

Usage:
    kmeans.py train <input_path> [<checkpoint_id>] [-n=<num>]
    kmeans.py encode <checkpoint_id> <input_path> <output_path>

Options:
    -n=<num>    Number of clusters to consider [default: 256]
"""

import docopt
import numpy as np
import os
import scipy.io
import scipy.cluster
import time

from utils.shapes_dataset import load_or_generate_dataset
CKPT_FORMAT = './checkpoints/{id}/checkpoint_{id}.ckpt'
ID_ = str(time.time())[-6:]


def main():
    """Runs main training and testing script."""
    arguments = docopt.docopt(__doc__, version='Kmeans 1.0')
    checkpoint_id = arguments['<checkpoint_id>']
    checkpoint_path = CKPT_FORMAT.format(id=checkpoint_id or ID_)
    data = load_or_generate_dataset(arguments['<input_path>'])
    data['X'] = data['X'].reshape((-1, np.prod(data['X'].shape[1:]))).astype(
        np.float32)
    if arguments['train']:
        centroids = train(data, int(arguments['-n']))
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        print(' * Saved model to %s' % checkpoint_path)
        scipy.io.savemat(checkpoint_path, {'centroids': centroids})
    else:
        centroids = scipy.io.loadmat(checkpoint_path)['centroids']
        data = featurize_w_centroids(data, centroids)
        scipy.io.savemat(arguments['<output_path>'], data)
        print(' * Saved featurized data to', arguments['<output_path>'])


def train(data, num_clusters):
    """Train by running k-means clustering.

    :param data: Data containing X and Y
    :param num_clusters: Number of clusters to consider
    :return: List of centroids
    """
    codebook, distortion = scipy.cluster.vq.kmeans(data['X'], num_clusters)
    return list(codebook)


def featurize_w_centroids(data, centroids):
    """Compute new featurization of data using centroids.

    Each sample's new featurization is its distance to each of the centroids.

    :param data: Data containing X and Y
    :param centroids: List of centroids
    :return: Data with featurized X and Y
    """
    Xhat = None
    for sample in data['X']:
        x = np.array([sample.dot(centroid) for centroid in centroids])
        Xhat = x if Xhat is None else np.vstack((Xhat, x))
    data['X'] = Xhat
    return data


if __name__ == '__main__':
    main()
