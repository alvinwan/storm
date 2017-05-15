"""Voxelize point cloud data.

Usage: voxelize.py
"""

import functools
import time

import numpy as np
import scipy.io
import sklearn.model_selection

import util

names = ['t', 'intensity', 'id',
         'x', 'y', 'z',
         'azimuth', 'range', 'pid']

formats = ['int64', 'uint8', 'uint8',
           'float32', 'float32', 'float32',
           'float32', 'float32', 'int32']

binType = np.dtype(dict(names=names, formats=formats))


def timed(f):
    """Decorator to time functions"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = f(*args, **kwargs)
        end = time.time()
        print("Time (%s):" % f.__name__, end - start)
        return rv
    return wrapper


def voxelize(
        shape=(20, 20, 20),
        datapath='../data/objects/excavator.0.10974.bin',
        threshold=0,  # This is a number between 0 and 100
        frame=1,
        cat=3,
        data=None):
    """Voxelize point cloud data, transforming a set of cartesian coordinates
    to a 3-dimensional tensor containing 0s and 1s.

    >>> V = voxelize((5, 5, 5),
    ...              data=np.array([[1, 1, 1], [2, 2, 2], [10, 10, 10]]))
    >>> V[0][0][0]
    1
    >>> V[1][1][1]
    1
    >>> np.sum(V)
    2
    """
    if data is None:
        # 3D points, one per row, from MAT file with 'x', 'y', and 'z' keys
        data = scipy.io.loadmat(datapath)
        P = np.hstack([data['xc'], data['yc'], data['zc'], data['cat'], data['frame']])
        P = P[P[:,4] == frame]  # Filter on frame
        P = P[(P[:,3] == 1) | (P[:,3] == 3)]  # Filter on cat
    else:
        P = data

    # Visualize as 2D image
#    plt.plot(P_3[:,0], P_3[:,2], 'r.')
#    plt.plot(P_4[:,0], P_4[:,2], 'b.')

    def bin_for_dim(P, dim, offset=1.25):
        mean = (min(P[:, dim]) + max(P[:, dim])) / 2
        return (mean + offset, mean - offset)

    voxel = np.zeros(shape, dtype=int)
    bins = [bin_for_dim(P, 0), bin_for_dim(P, 1), bin_for_dim(P, 2, offset=700)]

    for j, point in enumerate(P):
        x, y, z = (int((shape[i] - 1) * (point[i] - low) // (high - low))
                   for i, (high, low) in enumerate(bins))
        try:
            if threshold:
                voxel[x][y][z] += 1
            else:
                voxel[x][y][z] = min(voxel[x][y][z] + 1, 1)
        except IndexError:
            pass  # Ignore out-of-bounds voxels

    if threshold:
        boundary = np.percentile(voxel, threshold)
        voxel = np.where(voxel >= boundary, np.ones(shape, dtype=int), np.zeros(shape, dtype=int))
    return voxel


@timed
def timed_voxelize(*args, **kwargs):
    return voxelize(*args, **kwargs)


if __name__ == '__main__':
    num_frames = 176
    dims = 30  # Number of voxels per frame is this cubed
    data_file = '../1_zc-CropROIs-average-match.bin.mat'
    labels_file = '../../data/1_zc-CropROIs-average-match.labels'

    labels = []
    for frame, line in enumerate(open(labels_file).readlines()[1:], start=1):
        good, bad = line.split(',')[3:5]
        if good == '1':
            labels.append(1)
        elif bad == '1':
            labels.append(0)
        else:
            labels.append(2)
    print(len(labels))

    data = np.array([voxelize(shape=(dims, dims, dims), datapath=data_file, frame=frame, cat=(1, 3), threshold=99.1) for frame in range(1, num_frames + 1)])
    assert len(labels) == len(data)

    # Save copy with all data, in order
    scipy.io.savemat('../../data/new_all_molecules_%d.mat' % dims, {'X': data, 'Y': labels})

    # Create train/test splits using clathrin voxels and write to file
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.4, random_state=int(sum(map(ord, 'alvin'))))
    scipy.io.savemat('../../data/new_train_molecules_%d.mat' % dims, {'X': X_train, 'Y': Y_train})
    scipy.io.savemat('../../data/new_test_molecules_%d.mat' % dims, {'X': X_test, 'Y': Y_test})
