import copy
import numpy as np
import os
from cunet.train.config import config
import logging
from glob import glob
import gc
from joblib import Parallel, delayed
from cunet.preprocess.config import config as config_pre


logger = logging.getLogger('tensorflow')


def get_name(txt):
    return os.path.basename(os.path.normpath(txt)).replace('.npz', '')


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data, c_max=1):
    if c_max != 1:
        factor = np.divide(complex_max(data), c_max)
    else:
        factor = 1
    # normalize between 0-1
    output = np.divide((data - complex_min(data)),
                       (complex_max(data) - complex_min(data)))
    return np.multiply(output, factor)  # scale to the original range


def get_max_complex(data, keys):

    for key in keys:
        pos = np.argmax([np.abs(complex_max(data[key].item()[str(i)])) for i in data[key].item()])

    for key in keys:
        max_comp = np.array([complex_max(data[key].item()[str(i)]) for i in data[key].item()])[pos]

    return max_comp


def load_a_file(fl):
    data = {}
    print('Loading the file %s' % fl)
    data_tmp = np.load(fl, allow_pickle=True)
    sources = copy.deepcopy(data_tmp.files)
    sources.remove('config')
    c_max = get_max_complex(data_tmp, sources)

    for value in sources:
        data[value] = {}
        for i in data_tmp[value].item():
            data[value][i] = normlize_complex(data_tmp[value].item()[str(i)], c_max)
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


def get_data():
    return load_data(glob(os.path.join(config_pre.PATH_SPEC, '*i.npz')))
