import os
import re
import sys
from glob import glob
import shutil
import random
import tqdm
import numpy as np

def get_splits_once_separated(PATH_BASE):
    train_mix_wav_list = glob.glob(os.path.join(PATH_BASE, 'train', 'raw_audio', '*_mix.wav'))
    train_track_names = [x.split('/')[-1].replace('_mix.wav', '') for x in train_mix_wav_list]
    train = list(np.unique(['_'.join(x.split('_')[:-1]) for x in train_track_names]))

    test_mix_wav_list = glob.glob(os.path.join(PATH_BASE, 'test', 'raw_audio', '*_mix.wav'))
    test_track_names = [x.split('/')[-1].replace('_mix.wav', '') for x in test_mix_wav_list]
    test = list(np.unique(['_'.join(x.split('_')[:-1]) for x in test_track_names]))

    return train, test


def get_splits(PATH_BASE):
    mix_wav_list = glob.glob(os.path.join(PATH_BASE, 'raw_audio', '*_mix.wav'))
    track_names = [x.split('/')[-1].replace('_mix.wav', '') for x in mix_wav_list]
    track_names_no_ids = list(np.unique(['_'.join(x.split('_')[:-1]) for x in track_names]))
    train = random.sample(track_names_no_ids, int(len(track_names_no_ids)*0.66))
    test = [x for x in track_names_no_ids if x not in train]
    
    return train, test


def create_train_test_annotations(PATH_BASE):
    train, test = get_splits_once_separated(PATH_BASE)
    csv_list = glob.glob(os.path.join(PATH_BASE, 'train', 'f0s', '*.csv'))
    test_csv_files = []
    for i in test:
        test_csv_files += [x for x in csv_list if i in x]
    for file in tqdm.tqdm(test_csv_files):
        shutil.move(file, file.replace('train', 'test'))


def create_train_test(PATH_BASE):
    train, test = get_splits(PATH_BASE)
    wav_files = glob.glob(os.path.join(PATH_BASE, 'raw_audio', '*.wav'))
    train_wav_files = []
    for i in train:
        train_wav_files.append([x for x in wav_files if i in x])
    train_wav_files = [wav_file for sublist in train_wav_files for wav_file in sublist]
    test_wav_files = [x for x in wav_files if x not in train_wav_files]
    
    for file in tqdm.tqdm(train_wav_files):
        shutil.move(file, file.replace('raw_audio', 'raw_audio/train'))
    for file in tqdm.tqdm(test_wav_files):
        shutil.move(file, file.replace('raw_audio', 'raw_audio/test'))


def organize(PATH_BASE):
    train, test = get_splits_once_separated(PATH_BASE)
    '''
    for i in train:
        os.mkdir(os.path.join(PATH_BASE, 'train', 'raw_audio', str(i)))
        os.mkdir(os.path.join(PATH_BASE, 'train', 'f0s', str(i)))
    for i in test:
        os.mkdir(os.path.join(PATH_BASE, 'test', 'raw_audio', str(i)))
        os.mkdir(os.path.join(PATH_BASE, 'test', 'f0s', str(i)))
    '''

    train_wav_files = glob.glob(os.path.join(PATH_BASE, 'train', 'raw_audio', '*.wav'))
    test_wav_files = glob.glob(os.path.join(PATH_BASE, 'test', 'raw_audio', '*.wav'))

    train_csv_files = glob.glob(os.path.join(PATH_BASE, 'train', 'f0s', '*.csv'))
    test_csv_files = glob.glob(os.path.join(PATH_BASE, 'test', 'f0s', '*.csv'))

    for song in tqdm.tqdm(train):
        for i in tqdm.tqdm(train_wav_files):
            if song in i:
                shutil.move(i, i.replace('raw_audio', 'raw_audio/'+song))
        for i in tqdm.tqdm(train_csv_files):
            if song in i:
                shutil.move(i, i.replace('f0s', 'f0s/'+song))

    for song in tqdm.tqdm(test):
        for i in tqdm.tqdm(test_wav_files):
            if song in i:
                shutil.move(i, i.replace('raw_audio', 'raw_audio/'+song))
        for i in tqdm.tqdm(test_csv_files):
            if song in i:
                shutil.move(i, i.replace('f0s', 'f0s/'+song))

def reorganize_f0(PATH_BASE):
    train_directories = glob(os.path.join(PATH_BASE, 'train', 'f0s', '*'))
    f0s_train_files = []
    for i in train_directories:
        f0s_train_files = f0s_train_files + glob(os.path.join(i, '*.csv'))
    
    test_directories = glob(os.path.join(PATH_BASE, 'test', 'f0s', '*'))
    f0s_test_files = []
    for i in test_directories:
        f0s_test_files = f0s_test_files + glob(os.path.join(i, '*.csv'))

    for i in (f0s_train_files + f0s_test_files):
        song_name = i.split('/')[-2]
        shutil.move(i, i.replace(song_name + '/', ''))

    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    reorganize_f0('/mnt/md1/genis/Saraga-SS-Synth/')

