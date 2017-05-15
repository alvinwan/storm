"""Utilities for the shape dataset and visualization."""
import itertools
from sys import platform

import matplotlib
if platform == 'darwin':
    matplotlib.use('macosx')
if 'linux' in platform:
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import os.path
import scipy.io


#############
# CONSTANTS #
#############

def shapenet_data(prefix='../'):
    train_data = scipy.io.loadmat(
        os.path.join(prefix, 'data/train_shapenet.mat'))
    test_data = scipy.io.loadmat(
        os.path.join(prefix, 'data/test_shapenet.mat'))
    X, Y = train_data['X'], train_data['Y']
    X_test, Y_test = test_data['X'], test_data['Y']
    return (X, Y), (X_test, Y_test)


##################
# TRAINING UTILS #
##################

def value_to_summary(value, tag):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def load_data(train=None, test=None, conv=False, shape=None):
    """
    Load data per command line arguments.

    :param train: Data to train on, labels ignored.
    :param train: Train data to encode, must have labels.
    :param test: Test data to encode, must have labels.
    :return: (X, Y), (X_test, Y_test)
    """
    # Data loading and preprocessing
    if train is None:
        print(' * [INFO] Loading shapenet data...')
        (X, Y), (X_test, Y_test) = shapenet_data()
    else:
        print(' * [INFO] Loading %s data...' % train)
        if train.endswith('.npy'):
            train_data = np.load(train)
            X, Y = train_data[:, :-1], train_data[:, -1]
        else:
            train_data = scipy.io.loadmat(train)
            X, Y = train_data['X'], train_data['Y']
        if test is not None and test.endswith('.npy'):
            test_data = np.load(test)
            X, Y = test_data[:, :-1], test_data[:, -1]
        elif test is not None:
            test_data = scipy.io.loadmat(test)
            X_test, Y_test = test_data['X'], test_data['Y']
    if not shape:
        shape = [-1] + ([np.prod(X.shape[1:])] if not conv else [30, 30, 30, 1])
    X = X.reshape(shape)
    Y = Y.reshape((-1, 1))
    if test is None:
        return X, Y
    X_test = X_test.reshape(shape)
    return (X, Y), (X_test, Y_test)


####################
# SHAPE GENERATORS #
####################


def norms(radius, norm, dim=12):
    """Generates a shape including all {x : |x|_norm < radius}.

    :param radius: Radius of shape.
    :param norm: Norm to use.
    :param dim: Dimensions of cuboid to generate all shapes in.
    :return: List of tensors, representing the normed-shapes.
    """
    shape = np.zeros([dim, dim, dim])
    center = np.array([dim/2, dim/2, dim/2])
    for x in range(0, dim):
        for y in range(0, dim):
            for z in range(0, dim):
                distance = np.linalg.norm(
                    center - np.array([x, y, z]), ord=norm)
                if distance <= radius:
                    shape[x][y][z] = 1
    return shape


def diamond(length, dim=12):
    return norms(length, 1, dim)


def sphere(radius, dim=12):
    return norms(radius, 2, dim)


def cube(side_len, dim=12):
    cube = np.zeros([dim, dim, dim])
    for x in range(dim//2 - side_len//2, dim//2 + side_len//2):
        for y in range(dim//2 - side_len//2, dim//2 + side_len//2):
            for z in range(dim//2 - side_len//2, dim//2 + side_len//2):
                cube[x][y][z] = 1
    return cube


def noise(dim=12, prob=0.99):  # Higher prob means more sparse
    assert 0 <= prob <= 1
    return np.random.choice([0,1], size=(dim, dim, dim), p=[prob, 1 - prob])


#################
# VISUALIZATION #
#################


def get_3d_axis(dim=12, elev=None, azim=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.view_init(elev, azim)

    plt.xlim(0, dim)
    plt.ylim(0, dim)
    ax.set_zlim(0, dim)

    plt.xticks(np.arange(0, dim, 2))
    plt.yticks(np.arange(0, dim, 2))
    ax.set_zticks(np.arange(0, dim, 2))

    return ax


# http://stackoverflow.com/a/42611693
def plot3d(matrix, ax=None, color='b', elev=None, azim=None, save=None, show=False):
    assert matrix.shape[0] == matrix.shape[1] == matrix.shape[2]
    if not ax:
        ax = get_3d_axis(matrix.shape[0], elev, azim)
    def plot_cube(pos=(0,0,0),ax=None):
        if ax !=None:
            X, Y, Z = _cuboid_data(pos)
            ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i,j,k] >= 0.5:
                    # plot_cube(pos=(i-0.5,j-0.5,k-0.5), ax=ax)
                    plot_cube(pos=(i+0.5, j+0.5, k+0.5), ax=ax)
    if save:
        if '.' not in save:
            save += '.png'
        plt.savefig(save)
    if show:
        plt.show()
    return ax


# http://stackoverflow.com/a/42611693
def _cuboid_data(pos, size=(1,1,1)):
    o = [a - b / 2 for a, b in zip(pos, size)]
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return x, y, z


def scatter(xs, ys, zs, save=None):
    plt.figure()
    # Fix memory issue
    matplotlib.pyplot.rcParams['agg.path.chunksize'] = 20000
    # Normalize zs to 0-1 range
    zs = (zs - np.min(zs)) / np.max(zs)
    plt.scatter(xs, ys, c=zs, cmap=plt.cm.Blues)
    if save:
        if '.' not in save:
            save += '.png'
        plt.savefig(save)


############################
# CONFUSION MATRIX UTILITY #
############################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    plot3d(norms(6, 1.5, 14))
    plt.show()
    plot3d(noise(20))  # Random noise in 20x20x20 space
    plt.show()
    plot3d(cube(4, 8))  # 4x4x4 cube in 8x8x8 space
    plt.show()
