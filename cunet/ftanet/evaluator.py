import re
import numpy as np
import pandas as pd
import numpy
import glob
from tqdm import tqdm
import random
import pickle
from numpy.core.fromnumeric import std
import mir_eval
from cfp import cfp_process
from tensorflow import keras

import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import categorical_accuracy
from loader import load_data_for_test, load_data

from tensorflow.keras.models import load_model

from constant import *
from loader import *

from network.ftanet import create_model
from loader import get_CenFreq


def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)


def est(output, CenFreq, time_arr):
    # output: (freq_bins, T)
    CenFreq[0] = 0
    est_time = time_arr
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]

    if len(est_freq) != len(est_time):
        new_length = min(len(est_freq), len(est_time))
        est_freq = est_freq[:new_length]
        est_time = est_time[:new_length]

    est_arr = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)

    return est_arr


def iseg(data):
    # data: (batch_size, freq_bins, seg_len)
    new_length = data.shape[0] * data.shape[-1]  # T = batch_size * seg_len
    new_data = np.zeros((data.shape[1], new_length))  # (freq_bins, T)
    for i in range(len(data)):
        new_data[:, i * data.shape[-1] : (i + 1) * data.shape[-1]] = data[i]
    return new_data


def get_est_arr(model, x_list, y_list, batch_size):
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j * batch_size:]
                length = x.shape[0] - j * batch_size
            else:
                X = x[j * batch_size: (j + 1) * batch_size]
                length = batch_size
            
            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k])
            prediction = model.predict(X, length)
            preds.append(prediction)
        
        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)
        
        # ground-truth
        
        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        # CenFreq = get_CenFreq(StartFreq=20, StopFreq=2048, NumPerOct=60)
        # CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=111)
        # CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=190)
        est_arr = est(preds, CenFreq, y)
        
    # VR, VFA, RPA, RCA, OA
    return est_arr


def get_pitch_track(filename):
    print('Loading model...')
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights(
        filepath='./model/baseline/OA/best_OA'
    ).expect_partial()
    print('Model loaded!')
    
    xlist = []
    timestamps = []
    # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
    feature, _, time_arr = cfp_process(filename, sr=8000, hop=80)
    print('feature', np.shape(feature))
    
    data = batchize_test(feature, size=128)
    xlist.append(data)
    timestamps.append(time_arr)
    
    estimation = get_est_arr(model, xlist, timestamps, batch_size=16)

    return estimation[:, 0], estimation[:, 1]


def save_pitch_track_to_dataset(filename, est_time, est_freq):
    # Write txt annotation to file
    with open(filename, 'w') as f:
        for i, j in zip(est_time, est_freq):
            f.write("{}, {}\n".format(i, j))
    print('Saved with exit to {}'.format(filename))


def select_vocal_track(ypath, lpath):
    ycsv = pd.read_csv(ypath, names=["time", "freq"])
    gt0 = ycsv['time'].values
    gt0 = gt0[:, np.newaxis]
    
    gt1 = ycsv['freq'].values
    gt1 = gt1[:, np.newaxis]
    
    z = np.zeros(gt1.shape)
    
    f = open(lpath, 'r')
    lines = f.readlines()
    
    for line in lines:
        
        if 'start_time' in line.split(',')[0]:
            continue
        st = float(line.split(',')[0])
        et = float(line.split(',')[1])
        sid = line.split(',')[2]
        for i in range(len(gt1)):
            if st < gt0[i, 0] < et and 'singer' in sid:
                z[i, 0] = gt1[i, 0]
    
    gt = np.concatenate((gt0, z), axis=1)
    return gt
 

def get_files_to_test(fp, artist, artists_to_track_mapping):
    # Get track to train
    tracks_to_test = artists_to_track_mapping[artist]

    # Get filenames to train
    files_to_test = []
    for track in tracks_to_test:
        files_to_test.append(fp + 'audio/' + track + '.wav')
    
    return files_to_test


if __name__ == '__main__':
    fp_synth = '/home/genis/Saraga-Melody-Synth/'
    fp_hindustani = '/home/genis/Hindustani-Synth-Dataset/'
    fp_medley = '/mnt/sda1/genis/carnatic_melody_dataset/resources/medley_aux/'
    fp_western_synth = '/home/genis/Western-Synth-Dataset_2/'
    
    #dataset_filelist_nosynth = glob.glob(fp_nosynth + 'audio/*.wav')
    #with open(fp_nosynth + 'artists_to_track_mapping.pkl', 'rb') as map_file:
    #    artists_to_track_mapping_nosynth = pickle.load(map_file)

    dataset_filelist_synth = glob.glob(fp_synth + 'audio/*.wav')
    with open(fp_synth + 'artists_to_track_mapping.pkl', 'rb') as map_file:
        artists_to_track_mapping = pickle.load(map_file)
    
    mahati_test = get_files_to_test(fp_synth, 'Mahati', artists_to_track_mapping)
    sumithra_test = get_files_to_test(fp_synth, 'Sumithra Vasudev', artists_to_track_mapping)
    modhumudi_test = get_files_to_test(fp_synth, 'Modhumudi Sudhakar', artists_to_track_mapping)
    chertala_test = get_files_to_test(fp_synth, 'Cherthala Ranganatha Sharma', artists_to_track_mapping)
    test_carnatic_list = [mahati_test, sumithra_test, modhumudi_test, chertala_test]

    test_files = []
    for i in test_carnatic_list:
        test_files = test_files + random.sample(i, 50)

    #carnatic_synth = test_model(test_files)

    '''
    mahati_test_synth = get_files_to_test(fp_synth, 'Mahati', artists_to_track_mapping_synth)
    sumithra_test_synth = get_files_to_test(fp_synth, 'Sumithra Vasudev', artists_to_track_mapping_synth)
    modhumudi_test_synth = get_files_to_test(fp_synth, 'Modhumudi Sudhakar', artists_to_track_mapping_synth)
    chertala_test_synth = get_files_to_test(fp_synth, 'Cherthala Ranganatha Sharma', artists_to_track_mapping_synth)
    test_carnatic_list = [mahati_test_synth, sumithra_test_synth, modhumudi_test_synth, chertala_test_synth]
    
    test_files = []
    for i in test_carnatic_list:
        test_files = test_files + random.sample(i, 50)
    
    medley_tracks = glob.glob(fp_medley + 'audio/*.wav')
    
    hindustani_testing_files = [
        'Raag_Kedar_43.wav',
        'Raag_Kedar_10.wav',
        'Raag_Kalyan_61.wav',
        'Raag_Kalyan_39.wav',
        'Raag_Kalyan_67.wav',
        'Raag_Kalyan_61.wav',
        'Raag_Kedar_20.wav',
        'Raag_Kedar_30.wav',
        'Raag_Kedar_43.wav',
        'Raag_Jog_47.wav',
        'Raag_Jog_37.wav',
        'Raag_Jog_29.wav',
        'Raag_Jog_18.wav',
        'Raag_Jog_12.wav',
        'Raag_Jog_2.wav',
        'Raag_Saraswati_6.wav',
        'Raag_Saraswati_25.wav',
        'Raag_Bhimpalasi_23.wav',
        'Raag_Bhimpalasi_33.wav',
        'Raag_Bhimpalasi_43.wav',
        'Raag_Bhimpalasi_45.wav',
        'Raag_Bhimpalasi_50.wav',
        'Raag_Shree_66.wav',
        'Raag_Dhani_8.wav',
        'Raag_Dhani_25.wav',
        'Raag_Dhani_33.wav',
        'Raag_Dhani_58.wav',
        'Raag_Dhani_35.wav',
        'Raag_Dhani_57.wav',
        'Raag_Bahar_44.wav',
        'Raag_Bahar_29.wav',
        'Multani_17.wav',
        'Raag_Rageshri_61.wav',
        'Raag_Rageshri_36.wav',
        'Maru_Bihag_4.wav',
        'Raageshree_10.wav',
        'Raageshree_12.wav',
        'Raag_Desh_9.wav',
        'Raag_Bhoopali_83.wav',
        'Bhairavi_Bhajan_6.wav',
        'Raag_Bairagi_22.wav',
        'Raag_Multani_11.wav',
        'Raga_Shree_-_Khayal_38.wav',
        'Todi_16.wav',
        'Todi_10.wav',
        'Todi_17.wav',
        'Todi_3.wav',
        'Sudh_Sarang_21.wav',
        'Sudh_Kalyan_25.wav',
        'Raga_Shree_-_Khayal_96.wav',
        'Raga_Lalit_-_Khayal_76.wav',
        'Raag_Yaman_20.wav',
        'Raag_Sooha_Kanada_28.wav',
        'Raag_Sohani_15.wav',
        'Raag_Shree_81.wav',
        'Raag_Sawani_16.wav',
        'Raag_Ramdasi_Malhar_15.wav',
        'Raag_Puriya_55.wav',
        'Raag_Poorva_96.wav',
        'Raag_Poorva_77.wav',
        'Raag_Paraj_25.wav',
        'Raag_Multani_74.wav',
        'Raag_Megh_41.wav',
        'Raag_Malkauns_71.wav',
        'Raag_Lalita_Gauri_24.wav',
        'Raag_Bhoopali_11.wav',
        'Raag_Bhimpalasi_56.wav',
        'Raag_Bibhas_24.wav',
        'Raag_Bihag_11.wav',
        'Raag_Bihag_27.wav',
        'Raag_Bhatiyar_7.wav',
        'Raag_Ahir_Bhairon_58.wav',
        'Raag_Ahir_Bhairon_3.wav',
        'Nirgun_Bhajan_15.wav',
        'Nat_Bhairon_8.wav',
        'Multani_0.wav',
        'Malkauns_1.wav',
        'Malkauns_3.wav',
        'Malkauns_10.wav',
        'Kalavati_10.wav',
        'Aahir_Bhairon_16.wav',
    ]
    '''
    
    hindustani_testing_files = glob.glob(fp_hindustani + 'audio/*.wav')
    #hindustani_testing_filenames = [fp_hindustani + 'audio/' + x for x in hindustani_testing_files]

    testing_files = [x for x in hindustani_testing_files if 'Deepki' in x] + \
                    [x for x in hindustani_testing_files if 'Raag_Jog' in x] + \
                    [x for x in hindustani_testing_files if 'Raag_Dhani' in x] + \
                    [x for x in hindustani_testing_files if 'Todi' in x] + \
                    [x for x in hindustani_testing_files if 'Malkauns' in x] + \
                    [x for x in hindustani_testing_files if 'Piloo' in x]
    hindustani_synth = test_model(testing_files)

    '''
    #mahati = test_model(mahati_test)
    #sumithra = test_model(sumithra_test, 'sumithra_nosynth')
    #modhumudi = test_model(modhumudi_test, 'modhumudi_nosynth')
    #chertala = test_model(chertala_test, 'chertala_nosynth')
    #mahati_synth = test_model(mahati_test_synth, 'mahati')
    #sumithra_synth = test_model(sumithra_test_synth, 'sumithra')
    #modhumudi_synth = test_model(modhumudi_test_synth, 'modhumudi')
    #carnatic_synth = test_model(test_files, 'carnatic')

    #hindustani_synth = test_model(hindustani_testing_files, 'hindustani')
    #carnatic_synth = test_model(test_files, 'carnatic')
    #scores_synth = test_model_on_medley(medley_tracks)

    adc_synth_files = [x for x in files_to_test if 'daisy' in x] + \
                      [x for x in files_to_test if 'pop' in x] + \
                      [x for x in files_to_test if 'opera' in x]
    
    mirex05_synth_files = [x for x in files_to_test if 'train' in x]
    #medley_synth_files = [x for x in files_to_test if x not in adc_synth_files and 'train' not in x]
    adc_scores = test_model(adc_synth_files)
    mirex_scores = test_model(mirex05_synth_files)
    #medley_scores = test_model(medley_synth_files)

    #scores_western_synth = test_model(files_to_test)
    adc_filelist = glob.glob(
        '/home/genis/FTANet-melodic/eval_datasets/ADC2004/*.wav'
    )
    scores_adc = test_model_on_ADC(adc_filelist)
    mirex05_filelist = glob.glob(
        '/home/genis/FTANet-melodic/eval_datasets/MIREX05/*.wav'
    )
    scores_mirex05 = test_model_on_MIREX05(mirex05_filelist)

    print('Mahati:', mahati_synth)
    print('Sumithra:', sumithra_synth)
    print('Modhmudi:', modhumudi_synth)
    print('Chertala:', chertala_synth)
    print('Complete carnatic:', carnatic_synth)
    print('Hindustani:', hindustani_synth)
    '''

