"""Downsample or upsample voxelized data.

The following script expects a matlab file with the provided key.

Usage:
    revoxelize.py <source> [options]

Options:
    --key=<key>         Key containing matrix in question in data. [default: X]
    --output=<out>      Output path [default: data/revoxelized.mat]
    --scale=<scale>     Scale of sampling [default: 2/3]
"""

import scipy.ndimage
import scipy.io
import docopt
import numpy as np


def revoxelize(data, scale=0.5):
    """Revoxelizes the provided data."""
    return np.array([scipy.ndimage.zoom(x, scale) for x in data])


def main():
    # arguments = docopt.docopt(__doc__, 'Revoxelize 1.0')
    # source = arguments['<source>']
    # scale = float(arguments['--scale'])
    # key = arguments['--key']
    # output = arguments['--output']
    source = 'data/train_shapenet.mat'
    scale, output, key = 2/3, 'data/train_shapenet_20x.mat', 'X'

    data = scipy.io.loadmat(source)
    data[key] = revoxelize(data[key], scale=scale)
    scipy.io.savemat(output, data)



if __name__ == '__main__':
    main()
