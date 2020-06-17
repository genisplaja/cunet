# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):
    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']

    GROUP = setting(
        default='simple_cnn', simple_dense='simple_dense',
        complex_dense='complex_dense',
        simple_cnn='simple_cnn', complex_cnn='complex_cnn',
    )
    PATH_BASE = '../data/satb_dst/models'
    NAME = '_satb_one_hot_f0s_360CQT_world_f0_global_x2'

    PATH_MODEL = setting(
        default=os.path.join(PATH_BASE, 'conditioned/simple_cnn'),
        standard=os.path.join(PATH_BASE, 'standard'),
        simple_dense=os.path.join(PATH_BASE, 'conditioned/simple_dense'),
        complex_dense=os.path.join(PATH_BASE, 'conditioned/complex_dense'),
        simple_cnn=os.path.join(PATH_BASE, 'conditioned/simple_cnn'),
        complex_cnn=os.path.join(PATH_BASE, 'conditioned/complex_cnn')
    )
    PATH_AUDIO = '../data/satb_dst/test_usecase2/complex'
    PATH_AUDIO_PRED = os.path.join(PATH_MODEL.default,NAME,'pred_audio_usecase2')
    TARGET = ['soprano', 'alto', 'tenor', 'bass']  # ['vocals', 'bass', 'bass_vocals'] -> not ready yet for complex conditions
    INSTRUMENTS = ['soprano', 'alto', 'tenor', 'bass']  # to check that has the same order than the training
    OVERLAP = 0
    MODE = setting(default='conditioned', standard='standard')
    EMB_TYPE = setting(
        default='cnn', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
