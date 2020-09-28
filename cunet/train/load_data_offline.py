import copy
import numpy as np
import os
from cunet.train.config import config
import logging
from glob import glob
import gc
from joblib import Parallel, delayed
from cunet.preprocess.config import config as config_pre
from matplotlib import pyplot as plt
import random


logger = logging.getLogger('tensorflow')


def get_name(txt):
    return os.path.basename(os.path.normpath(txt)).replace('.npz', '')


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data, c_max=1):
    """ Each source preserve its range of values i.e.
    if the max of source_1 is .89 the max in the mixture, after the normalizaiton
    it is still .89, not 1
    """
    if c_max != 1:
        factor = np.divide(complex_max(data), c_max)
    else:
        factor = 1
    # normalize between 0-1
    output = np.divide((data - complex_min(data)),
                       (complex_max(data) - complex_min(data)))
    return np.multiply(output, factor)  # scale to the original range


def get_max_complex(data, keys):
    # sometimes the max is not the mixture
    pos = np.argmax([np.abs(complex_max(data[i])) for i in keys])
    return np.array([complex_max(data[i]) for i in keys])[pos]


def load_a_file(fl):
    data = {}
    print('Loading the file %s' % fl)
    data_tmp = np.load(fl, allow_pickle=True)
    sources = copy.deepcopy(data_tmp.files)
    sources.remove('config')
    c_max = get_max_complex(data_tmp, sources)

    for value in sources:
        data[value] = normlize_complex(data_tmp[value], c_max)
    return (get_name(fl), data)


def load_data(files):
    """The data is loaded in memory just once for the generator to have direct
    access to it"""
    data = {
        k: v for k, v in Parallel(n_jobs=16, verbose=5)(
                delayed(load_a_file)(fl=fl) for fl in files
            )
    }
    _ = gc.collect()

    return data


def get_data(path=config_pre.PATH_SPEC):
    return load_data(glob(os.path.join(path, '*.npz')))

def get_indexes(path=config.INDEXES_TRAIN):
    indexes = {}
    print('Loading index file %s' % path)
    data_tmp = np.load(path, allow_pickle=True)

    for song in data_tmp.files:
        indexes[song] = {}
        if not song == 'config':
            for part in data_tmp[song].item():
                indexes[song][part] = data_tmp[song].item()[part]
    return indexes
