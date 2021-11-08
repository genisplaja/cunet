import copy
import numpy as np
import os
import tensorflow as tf
from cunet.train.config import config
from cunet.train.load_data_offline import get_data
from cunet.train.others.val_files import VAL_FILES
import random
import logging
import time

DATA = get_data()
logger = logging.getLogger('tensorflow')


def check_shape(data):
    n = data.shape[0]
    if n % 2 != 0:
        n = data.shape[0] - 1
    return np.expand_dims(data[:n, :], axis=2)


def get_name(txt):
    return os.path.basename(os.path.normpath(txt)).replace('.npz', '')


# def progressive(data, conditions, dx, val_set):
#     output = copy.deepcopy(data)
#     if (
#         config.PROGRESSIVE and np.max(np.abs(data)) > 0
#         and random.sample(range(0, 4), 1)[0] == 0   # 25% of doing it
#         and not val_set
#     ):
#         p = random.uniform(0, 1)
#         conditions[dx] = conditions[dx]*p
#         output[:, :, dx] = output[:, :, dx]*p
#     return output[:, :, dx], conditions


# def yield_data(indexes, files, val_set):
#     conditions = np.zeros(1).astype(np.float32)
#     n_frames = config.INPUT_SHAPE[1]
#     for i in indexes:
#         if i[0] in files:
#             if len(i) > 2:
#                 conditions = i[2]
#             yield {'data': DATA[i[0]][:, i[1]:i[1]+n_frames, :],
#                    'conditions': conditions, 'val': val_set}


# def load_indexes_file(val_set=False):
#     if not val_set:
#         indexes = np.load(config.INDEXES_TRAIN, allow_pickle=True)['indexes']
#         r = list(range(len(indexes)))
#         random.shuffle(r)
#         indexes = indexes[r]
#         files = [i for i in DATA.keys() if i is not VAL_FILES]
#     else:
#         # Indexes val has no overlapp in the data points
#         indexes = np.load(config.INDEXES_VAL, allow_pickle=True)['indexes']
#         files = VAL_FILES
#     return yield_data(indexes, files, val_set)


# def get_data_aug(data_complex, target, conditions, val_set):
#     mixture = data_complex[:, :, -1]
#     # 25% of doing it a random point of another track
#     if not val_set and random.sample(range(0, 4), 1)[0] == 0 and config.AUG:
#         mixture = copy.deepcopy(target)
#         n_frames = config.INPUT_SHAPE[1]
#         for i in np.where(conditions == 0)[0]:
#             uid = random.choice([i for i in DATA.keys()])
#             tmp_data = DATA[uid]
#             frame = random.choice(
#                 [i for i in range(tmp_data.shape[1]-config.INPUT_SHAPE[1])]
#             )
#             mixture = np.sum(
#                 [tmp_data[:, frame:frame+n_frames, i], mixture], axis=0
#             )
#     return np.abs(mixture)


def SSSSBatchGenerator(valid=False):
    while True:

        out_shapes = {'mixture':np.zeros(config.INPUT_SHAPE), 
                      'target':np.zeros(config.INPUT_SHAPE), 
                      'conditions':np.zeros([config.Z_DIM, 2])}

        # Get rand song
        if not valid:
            randsong = random.choice([i for i in DATA.keys() if i is not VAL_FILES])
            vocal_data = DATA[randsong]['vocals']
            mixture_data = DATA[randsong]['mixture']
        else:
            randsong = random.choice(VAL_FILES)
            vocal_data = DATA[randsong]['vocals']
            mixture_data = DATA[randsong]['mixture']

        parts = list(vocal_data.keys())
        actual_part = random.choice(parts)

        start_frame = 0
        end_frame   = 0

        # Get Start and End samples. Pick random part to calculate start/end spl
        while start_frame == 0:
            try:
                start_frame = random.randint(0, vocal_data[actual_part].shape[1]-config.INPUT_SHAPE[1]) # This assume that all stems are the same length
            except Exception as e: 
                print('Exception:', e)
                pass

        end_frame   = start_frame+config.INPUT_SHAPE[1]
        print(start_frame, end_frame)

        # Scale down all the group chunks based off number of sources per group
        out_shapes['mixture'] = check_shape(mixture_data[actual_part][:,start_frame:end_frame])

        # Take vocal source as target
        got_target = False
        while got_target == False:
            try:
                target = 'vocals'
                out_shapes['conditions'] = np.load(config.INDEXES_TRAIN, allow_pickle=True)[randsong].item()['vocals'][actual_part][start_frame:end_frame,:]
                #out_shapes['conditions'] = np.argmax(onehot_f0s,axis=1)
                out_shapes['target'] = check_shape(vocal_data[actual_part][:,start_frame:end_frame])
                got_target = True
            except Exception as e: 
                print('Exception:', e)
                pass

        print(np.shape(out_shapes['mixture']))
        print(np.shape(out_shapes['conditions']))   
        print(np.shape(out_shapes['target']))

        yield out_shapes


def convert_to_estimator_input(d):
    # just the mixture standar mode
    inputs = tf.ensure_shape(d["mixture"], config.INPUT_SHAPE)
    if config.MODE == 'conditioned':
        if config.CONTROL_TYPE == 'dense':
            c_shape = (1, config.Z_DIM)
        if config.CONTROL_TYPE == 'cnn':
            c_shape = (config.Z_DIM, 1)
        cond = tf.ensure_shape(tf.reshape(d['conditions'], c_shape), c_shape)
        # mixture + condition vector z
        inputs = (inputs, cond)
        # target -> isolate instrument
    outputs = tf.ensure_shape(d["target"], config.INPUT_SHAPE)
    return (inputs, outputs)


def dataset_generator(val_set=False):
    ds = tf.data.Dataset.from_generator(
        SSSSBatchGenerator,
        {'mixture': tf.complex64, 'target': tf.complex64, 'conditions': tf.float32},
        args=[val_set]
    ).map(
        convert_to_estimator_input, num_parallel_calls=config.NUM_THREADS
    ).batch(
        config.BATCH_SIZE, drop_remainder=True
    ).prefetch(
        buffer_size=config.N_PREFETCH
    )
    if not val_set:
        ds = ds.repeat()
    return ds