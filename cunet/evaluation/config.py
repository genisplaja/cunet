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
    PATH_BASE = '/home/genis/cunet/resources/Saraga-SS-Synth/models'
    NAME = 'ssss_one_hot_f0s'

    PATH_MODEL = setting(
        default=os.path.join(PATH_BASE, 'conditioned/complex_cnn'),
        standard=os.path.join(PATH_BASE, 'standard'),
        simple_dense=os.path.join(PATH_BASE, 'conditioned/simple_dense'),
        complex_dense=os.path.join(PATH_BASE, 'conditioned/complex_dense'),
        simple_cnn=os.path.join(PATH_BASE, 'conditioned/simple_cnn'),
        complex_cnn=os.path.join(PATH_BASE, 'conditioned/complex_cnn')
    )
    PATH_AUDIO = '/home/genis/cunet/resources/Saraga-SS-Synth/test/complex'
    PATH_INDEXES = '/home/genis/cunet/resources/Saraga-SS-Synth/test/indexes'
    PATH_AUDIO_PRED = os.path.join(PATH_MODEL.default,NAME,'pred_audio')
    TARGET = ['vocals']
    INSTRUMENTS = ['vocals', 'mixture']  # to check that has the same order than the training
    OVERLAP = 0
    MODE = setting(default='conditioned', standard='standard')
    EMB_TYPE = setting(
        default='cnn', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
