# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import tensorflow as tf
import os
from cunet.preprocess.config import config as config_prepro


class config(Config):

    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']
    # General

    MODE = setting(default='conditioned', standard='standard')

    NAME = '_dummy'
    ADD_TIME = False    # add the time and date in the name
    TARGET = 'vocals'   # only for standard version

    # GENERATOR
    PATH_BASE = '../data/satb_dst/'
    # default = conditioned
    INDEXES_TRAIN = setting(
        default={
        'soprano':[1,0,0,0],
        'alto':   [0,1,0,0],
        'tenor':  [0,0,1,0],
        'bass':   [0,0,0,1]
        }
    )
    INDEXES_VAL = setting(
        default=os.path.join(
            PATH_BASE, ''),
        standard=os.path.join(
            PATH_BASE, '')
    )

    NUM_THREADS = 32#tf.data.experimental.AUTOTUNE   # 32
    N_PREFETCH = 4096#tf.data.experimental.AUTOTUNE  # 4096

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 1e-8
    EARLY_STOPPING_PATIENCE = 60
    REDUCE_PLATEAU_PATIENCE = 15

    # training
    BATCH_SIZE = 16
    N_BATCH = 2048
    N_EPOCH = 200
    PROGRESSIVE = True
    AUG = True
    USE_CASE = 1 # 0: max 1 singer, 1: exactly 1, 2: minimum 1

    # unet paramters
    INPUT_SHAPE = [512, 128, 1]  # freq = 512, time = 128
    FILTERS_LAYER_1 = 16
    N_LAYERS = 6
    LR = 1e-3
    ACTIVATION_ENCODER = 'leaky_relu'
    ACTIVATION_DECODER = 'relu'
    ACT_LAST = 'sigmoid'
    LOSS = 'mean_absolute_error'

    # -------------------------------

    # control parameters
    CONTROL_TYPE = setting(
        'cnn', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
    FILM_TYPE = setting(
        'complex', simple_dense='simple', complex_dense='complex',
        simple_cnn='simple', complex_cnn='complex'
    )
    Z_DIM = 4 # f0 point for each spec frame
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        1008, simple_dense=6, complex_dense=1008,
        simple_cnn=6, complex_cnn=1008
    )

    # cnn control
    N_FILTERS = setting(
        [32, 64, 256], simple_cnn=[16, 32, 64], complex_cnn=[32, 64, 256]
    )
    PADDING = ['same', 'same', 'valid']
    # Dense control
    N_NEURONS = setting(
        [16, 256, 1024], simple_dense=[16, 64, 256],
        complex_dense=[16, 256, 1024]
    )
