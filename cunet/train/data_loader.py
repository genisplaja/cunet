import copy
import numpy as np
import os
import tensorflow as tf
from cunet.train.config import config
from cunet.train.load_data_offline import (get_data, get_indexes)
from cunet.train.others.val_files import VAL_FILES
import random
import logging
import time
from matplotlib import pyplot as plt
import soundfile
import librosa
from cunet.preprocess.config import config as config_prepro

DATA_TRAIN = get_data()
DATA_VAL = get_data(path=os.path.join(config.PATH_BASE,'valid/complex'))
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


def SATBBatchGenerator(valid=False):

    counter = 0

    while True:

        sources = ['soprano','alto','tenor','bass']
        out_shapes = {'mixture':np.zeros((config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1)), 
                      'target':np.zeros((config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1)), 
                      'conditions':np.zeros((config.BATCH_SIZE,config.Z_DIM,1))}

        # Get rand song
        if not valid:
            DATA = DATA_TRAIN
        else:
            DATA = DATA_VAL

        randsong = random.choice([i for i in DATA.keys()])

        # Get all available part from chosen song
        part_count = DATA[randsong]

        for i in range(config.BATCH_SIZE):

            # Use-Case: At most one singer per part
            if (config.USE_CASE==0):
                max_num_singer_per_part = 1
                randsources = random.sample(sources, random.randint(1,len(sources)))                   # Randomize source pick if at most one singer per part
            # Use-Case: Exactly one singer per part
            elif (config.USE_CASE==1):
                max_num_singer_per_part = 1
                randsources = sources                                                                  # Take all sources + Set num singer = 1
            # Use-Case: At least one singer per part
            else:
                max_num_singer_per_part = 4
                randsources = sources                                                                  # Take all sources + Set max num of singer = 4 

            start_frame = 0
            end_frame   = 0

            # Get Start and End samples. Pick random part to calculate start/end spl
            while start_frame == 0:
                try:
                    randpart = random.choice(sources)
                    start_frame = random.randint(0,DATA[randsong][randpart]['1'].shape[1]-config.INPUT_SHAPE[1]) # This assume that all stems are the same length
                except Exception as e: 
                    print(e)
                    pass

            end_frame   = start_frame+config.INPUT_SHAPE[1]

            # Get Random Sources: 
            randsources_for_song = [] 
            for source in randsources:
                # If no singer in part, default it to one and fill array with zeros later
                singers_for_part = len(DATA[randsong][randpart].keys())
                if singers_for_part>0:
                    max_for_part = singers_for_part if singers_for_part < max_num_singer_per_part else max_num_singer_per_part
                else:
                    max_for_part = 1 

                num_singer_per_part = random.randint(1,max_for_part)                      # Get random number of singer per part based on max_for_part
                singer_num = random.sample(range(1,max_for_part+1),num_singer_per_part)   # Get random part number for the number of singer based off max_for_part
                randsources_for_part = np.repeat(source,num_singer_per_part)              # Repeat the parts according to the number of singer per group
                randsources_for_part = ["{}{}".format(a_, b_) for a_, b_ in zip(randsources_for_part, singer_num)] # Concatenate strings for part name
                randsources_for_song+=randsources_for_part

            # Retrieve the chunks and store them in output shapes 
            zero_source_counter = 0                                        
            for source in randsources_for_song:

                # Try to retrieve chunk. If part doesn't exist, create array of zeros instead
                try:
                    source_chunk = np.abs(DATA[randsong][source[:-1]][source[-1]][:,start_frame:end_frame]) # Retrieve part's chunk
                except:
                    print('No source found for: '+str(source))
                    zero_source_counter += 1
                    source_chunk = np.zeros(config.INPUT_SHAPE)

                out_shapes['mixture'][i] = np.add(out_shapes['mixture'][i],check_shape(source_chunk)) # Add the chunk to the mix
            
            # Scale down all the group chunks based off number of sources per group
            scaler = len(randsources_for_song) - zero_source_counter
            out_shapes['mixture'][i] = out_shapes['mixture'][i] / scaler

            # Take random source as target
            got_target = False
            while got_target == False:
                try:
                    target = random.choice(randsources_for_song)
                    out_shapes['conditions'][i] = np.expand_dims(config.INDEXES_TRAIN[target[:-1]],axis=1)
                    out_shapes['target'][i] = check_shape(np.abs(DATA[randsong][target[:-1]][target[-1]][:,start_frame:end_frame])) / scaler
                    got_target = True
                except Exception as e: 
                    print(e)
                    pass

        yield out_shapes


def convert_to_estimator_input(d):

    if config.CONTROL_TYPE == 'dense':
        c_shape = (config.BATCH_SIZE,1, config.Z_DIM)
    if config.CONTROL_TYPE == 'cnn':
        c_shape = (config.BATCH_SIZE,config.Z_DIM, 1)

    cond = tf.reshape(d['conditions'], c_shape)

    inputs = (d["mixture"], cond)
    outputs = d["target"]

    return (inputs, outputs)


def dataset_generator(val_set=False):

    out_shapes = {'mixture':(config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1), 
                'target':(config.BATCH_SIZE,config.INPUT_SHAPE[0],config.INPUT_SHAPE[1],1), 
                'conditions':(config.BATCH_SIZE,config.Z_DIM,1)}

    ds = tf.data.Dataset.from_generator(
        SATBBatchGenerator,
        output_types={'mixture': tf.float32, 'target': tf.float32, 'conditions': tf.float32},
        output_shapes=out_shapes,
        args=[val_set]
    ).map(
        convert_to_estimator_input#, num_parallel_calls=config.NUM_THREADS
    )#.batch(
    #     config.BATCH_SIZE, drop_remainder=True
    # )#.prefetch(
    #     buffer_size=config.N_PREFETCH
    # )
    if not val_set:
        ds = ds.repeat()
    return ds
