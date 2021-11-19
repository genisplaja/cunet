import copy
import math
import tqdm
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


def check_shape(data):
    n = data.shape[0]
    if n % 2 != 0:
        n = data.shape[0] - 1
    return np.expand_dims(data[:n, :], axis=2)


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


def batch_size(feat_list, features, indexes=False):
    frame_size = config.INPUT_SHAPE[1]
    if indexes is False:
        for i in np.arange(math.floor(features.shape[1]/frame_size)):
            feat_list.append(check_shape(features[:, i*frame_size:(i+1)*frame_size]))

    else:
        for i in np.arange(math.floor(features.shape[0]/frame_size)):
            feat_list.append(features[i*frame_size:(i+1)*frame_size, :])

    return feat_list


def get_data(path=config_pre.PATH_SPEC):
    data = load_data(random.sample(glob(os.path.join(path, '*.npz')), 25))
    indexes = np.load(config.INDEXES_TRAIN, allow_pickle=True)
    train_tracks = random.sample([i for i in data.keys()], 20)
    validation_tracks = [i for i in data.keys() if i not in train_tracks]

    mixture_list, target_list, conditions_list = [], [], []
    for i in tqdm.tqdm(train_tracks):
        song_conditions = indexes[i].item()['vocals']
        for part in data[i]['vocals']:
            mixture_list = batch_size(mixture_list, data[i]['mixture'][part])
            target_list = batch_size(target_list, data[i]['vocals'][part])
            conditions_list = batch_size(conditions_list, song_conditions[part], indexes=True)

    mixture_list_val, target_list_val, conditions_list_val = [], [], []
    for i in tqdm.tqdm(validation_tracks):
        song_conditions = indexes[i].item()['vocals']
        for part in data[i]['vocals']:
            mixture_list_val = batch_size(mixture_list, data[i]['mixture'][part])
            target_list_val = batch_size(target_list, data[i]['vocals'][part])
            conditions_list_val = batch_size(conditions_list, song_conditions[part], indexes=True)

    return mixture_list, target_list, conditions_list, mixture_list_val, target_list_val, conditions_list_val


def get_indexes(path=config.INDEXES_TRAIN):
    indexes = {}
    print('Loading index file %s' % path)
    data_tmp = np.load(path, allow_pickle=True)

    for song in data_tmp.files:
        indexes[song] = {}
        if not song == 'config':
            for part in data_tmp[song].item():
                indexes[song][part] = {}
                for part_num in data_tmp[song].item()[part].keys():
                    f0 = data_tmp[song].item()[part][part_num]
                    indexes[song][part][part_num] = f0
    return indexes

