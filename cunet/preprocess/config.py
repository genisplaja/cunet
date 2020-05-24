# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    groups = ['train', 'test']
    PATH_BASE = '../data/satb_dst/'
    PATH_RAW = setting(
        default=PATH_BASE+'test_usecase1/raw_audio',
        train=PATH_BASE+'train/raw_audio', test=PATH_BASE+'test_usecase2/raw_audio'
    )
    PATH_SPEC = setting(
        default=PATH_BASE+'test_usecase1/complex',
        train=PATH_BASE+'train/complex', test=PATH_BASE+'test_usecase2/complex'
    )
    PATH_INDEXES = setting(
        default=PATH_BASE+'test_usecase1/indexes',
        train=PATH_BASE+'train/indexes', test=PATH_BASE+'test_usecase2/indexes'
    )
    PATH_F0S = setting(
        default=PATH_BASE+'test_usecase1/f0s',
        train=PATH_BASE+'train/f0s', test=PATH_BASE+'test_usecase2/f0s'
    )
    FR = 22050
    FFT_SIZE = 1024
    HOP = 256
    MODE = 'conditioned'
    GROUP = 'test' # mainly used for spectrogram pre-processing: need to compute the mixture as well

    CQT_BINS = 360
    MIN_FREQ = 32.7
    BIN_PER_OCT = 60

    # SATB
    INTRUMENTS = ['soprano', 'alto', 'tenor', 'bass']
    CONDITIONS = ['soprano', 'alto', 'tenor', 'bass']
    
    CONDITION_MIX = 1       # complex conditions -> 1 only original instrumets, 2 combination of 2 instruments, etc
    ADD_ZERO = True         # add the zero condition
    ADD_ALL = True          # add the all mix condition
    ADD_IN_BETWEEN = 1.     # in between interval for the combination of several instruments
    STEP = 1                # step in frames for creating the input data
    CHUNK_SIZE = 4          # chunking the indexes before mixing -> define the number of data points of the same track
