import copy
import pdb
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from cunet.train.config import config
from cunet.train.load_data_offline import get_data
import random
import logging
import librosa
import tqdm

DATA_MIXTURE, DATA_TARGET, DATA_CONDITIONS, DATA_MIXTURE_VAL, DATA_TARGET_VAL, DATA_CONDITIONS_VAL = get_data()

print(np.shape(DATA_MIXTURE))
print(np.shape(DATA_TARGET))
print(np.shape(DATA_CONDITIONS))

print(np.shape(DATA_MIXTURE_VAL))
print(np.shape(DATA_TARGET_VAL))
print(np.shape(DATA_CONDITIONS_VAL))

logger = logging.getLogger('tensorflow')


def check_shape(data):
    n = data.shape[0]
    if n % 2 != 0:
        n = data.shape[0] - 1
    return np.expand_dims(data[:n, :], axis=2)

def istft(data):
    return librosa.istft(data)


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

'''
def SSSSBatchGenerator(valid=False):
    
    while True:

        #out_shapes = {'mixture':np.zeros((config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1)), 
        #              'target':np.zeros((config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1)), 
        #              'conditions':np.zeros((config.BATCH_SIZE,config.Z_DIM[0],config.Z_DIM[1]))}

        out_shapes = {'mixture':np.zeros((config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1)), 
                        'target':np.zeros((config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1)), 
                        'conditions':np.zeros((config.Z_DIM[0],config.Z_DIM[1]))}

        #out_shapes = {
        #    'mixture': [],
        #    'target': [],
        #    'conditions': []
        #}

        #for i in np.arange(config.BATCH_SIZE):
        
        #out_shapes = {
        #    'mixture': [],
        #    'target': [],
        #    'conditions': []
        #}

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

        start_frame = random.randint(0, vocal_data[actual_part].shape[1]-config.INPUT_SHAPE[1]) # This assume that all stems are the same length
        end_frame   = start_frame+config.INPUT_SHAPE[1]
        #print(start_frame, end_frame)

        # Scale down all the group chunks based off number of sources per group
        mixture_data = check_shape(mixture_data[actual_part][:,start_frame:end_frame])

        # Take vocal source as target
        condition_data = INDEXES[randsong].item()['vocals'][actual_part][start_frame:end_frame,:]
        #out_shapes['conditions'] = np.argmax(condition_data,axis=1)
        target_data = check_shape(vocal_data[actual_part][:,start_frame:end_frame])
        
        out_shapes['conditions'] = condition_data
        out_shapes['target'] = target_data
        out_shapes['mixture'] = mixture_data

        #print(np.shape(out_shapes['mixture']))
        #print(np.shape(out_shapes['conditions']))   
        #print(np.shape(out_shapes['target']))


        inputs = (out_shapes["mixture"], out_shapes["conditions"])
        outputs = out_shapes["target"]

        #yield out_shapes
        yield (inputs, outputs)
        

def convert_to_estimator_input(d):

    inputs = (d["mixture"], d["conditions"])
    outputs = d["target"]

    return (inputs, outputs)


def dataset_generator(val_set=False):


    #out_shapes = {'mixture':(config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1), 
    #            'target':(config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1), 
    #            'conditions':(config.BATCH_SIZE,config.Z_DIM[0],config.Z_DIM[1])}

    out_shapes = {'mixture':(config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1), 
                'target':(config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1), 
                'conditions':(config.Z_DIM[0],config.Z_DIM[1])}

    ds = tf.data.Dataset.from_generator(
        SSSSBatchGenerator,
        output_types={'mixture': tf.float32, 'target': tf.float32, 'conditions': tf.float32},
        output_shapes=out_shapes,
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
'''

def create_data_generator_train(batch_size):

    length = len(DATA_TARGET)
    idx = [i for i in range(length)]
    random.shuffle(idx)

    i = 0
    while True:

        if i + batch_size > length:
            i = 0
            random.shuffle(idx)

        # 每次取batch_size个key
        mixtures, targets, conditions = [], [], []
        for j in range(i, i + batch_size):
            # 对每一个，取feature
            mixture = DATA_MIXTURE[idx[j]]
            target = DATA_TARGET[idx[j]]
            condition = DATA_CONDITIONS[idx[j]]

            mixtures.append(mixture)
            targets.append(target)
            conditions.append(condition)

        i += batch_size
        yield ((np.stack(mixtures, axis=0), np.stack(conditions, axis=0)), np.stack(targets, axis=0))


def create_data_generator_val(batch_size):

    length = len(DATA_TARGET_VAL)
    idx = [i for i in range(length)]
    random.shuffle(idx)

    i = 0
    while True:

        if i + batch_size > length:
            i = 0
            random.shuffle(idx)

        # 每次取batch_size个key
        mixtures, targets, conditions = [], [], []
        for j in range(i, i + batch_size):
            # 对每一个，取feature
            mixture = DATA_MIXTURE_VAL[idx[j]]
            target = DATA_TARGET_VAL[idx[j]]
            condition = DATA_CONDITIONS_VAL[idx[j]]

            mixtures.append(mixture)
            targets.append(target)
            conditions.append(condition)

        i += batch_size
        yield ((np.stack(mixtures, axis=0), np.stack(conditions, axis=0)), np.stack(targets, axis=0))