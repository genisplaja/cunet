from glob import glob
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import mir_eval
import numpy as np
import os
import logging
import pandas as pd
from cunet.evaluation.config import config
from cunet.preprocess.config import config as config_prepro
from cunet.train.load_data_offline import normlize_complex
from cunet.preprocess.spectrogram import spec_complex
from cunet.train.models.FiLM_utils import FiLM_simple_layer
import soundfile as sf
from matplotlib import pyplot as plt
import random


logging.basicConfig(level=logging.INFO)

def plot_f0(data,name):

    plt.plot(np.argmax(data,axis=1))
    plt.savefig(os.path.join('./debug_fig','f0_'+name+'.png'))
    plt.clf()

def plot_spec(data,phase,name):

    data_rs = data#np.vstack((data, data[-1:, :]))
    plt.imshow(data_rs)
    print('Plot Spec:')
    print(np.shape(data))
    print(np.shape(data_rs))
    plt.savefig(os.path.join('./debug_fig','spec_'+name+'.png'))
    plt.clf()

    spec = reconstruct(data_rs, phase, None)

    sf.write(os.path.join('./debug_fig','audio_'+name+'.wav'),spec,config_prepro.FR)

def get_f0(song, group, part):

    f0 =  np.load(os.path.join(config_prepro.PATH_INDEXES,'indexes_SATB_F0s.npz') ,allow_pickle=True)[song].item()[group][part]
    return f0

def istft(data):
    return librosa.istft(
        data, hop_length=config_prepro.HOP,
        win_length=config_prepro.FFT_SIZE)

def save_pred_to_path(pred, name):
    sf.write(os.path.join(config.PATH_AUDIO_PRED,name+'.wav'), pred, config_prepro.FR)


def adapt_pred(pred, target):
    pred = (
        (np.max(target) - np.min(target))
        * ((pred - np.min(pred))/(np.max(pred) - np.min(pred)))
        + np.min(target)
    )
    pred += np.mean(target) - np.mean(pred)  # center_in_zero
    return pred
    # return (pred - np.mean(pred)) / np.std(target)


def reconstruct(pred_mag, orig_mix_phase, orig_mix_mag):
    pred_mag = pred_mag[:, :orig_mix_phase.shape[1]]
    pred_mag /= np.max(pred_mag)
    pred_spec = pred_mag * np.exp(1j * orig_mix_phase)
    return istft(pred_spec)


def prepare_a_song(spec, num_frames, num_bands, cond):
    size = spec.shape[1]

    segments = np.zeros(
        (size//(num_frames-config.OVERLAP)+1, num_bands, num_frames, 1),
        dtype=np.float32)

    segments_cond = np.zeros(
        (size//(num_frames-config.OVERLAP)+1, num_frames, config_prepro.CQT_BINS+1),
        dtype=np.float32)

    for index, i in enumerate(np.arange(0, size, num_frames-config.OVERLAP)):
        segment = spec[:num_bands, i:i+num_frames]
        segment_cond = cond[i:i+num_frames,:]
        tmp = segment.shape[1]

        if tmp != num_frames:
            segment = np.zeros((num_bands, num_frames), dtype=np.float32)
            segment_cond = np.zeros((num_frames,config_prepro.CQT_BINS+1), dtype=np.float32)
            segment[:, :tmp] = spec[:num_bands, i:i+num_frames]
            segment_cond[:tmp,:] = cond[i:i+num_frames,:]

        segments[index] = np.expand_dims(segment, axis=2)
        segments_cond[index] = segment_cond

    return segments, segments_cond


def separate_audio(path_audio, path_output, model, cond):
    y, _ = analize_spec(
        spec_complex(path_audio)['spec'], model, cond)
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    name = path_audio.split('/')[-1].replace('.mp3', '.wav')
    name = os.path.join(path_output, name)
    librosa.output.write_wav(name, y, sr=config_prepro.FR)
    return


def concatenate(data, shape):
    output = np.array([])
    if config.OVERLAP == 0:
        output = np.concatenate(data, axis=1)
    else:
        output = data[0]
        o = int(config.OVERLAP/2)
        f = 0
        if config.OVERLAP % 2 != 0:
            f = 1
        for i in range(1, data.shape[0]):
            output = np.concatenate(
                (output[:, :-(o+f), :], data[i][:, o:, :]), axis=1)
    if shape[0] % 2 != 0:
        # duplicationg the last bin for odd input mag
        output = np.vstack((output, output[-1:, :]))
    return output


def analize_spec(orig_mix_spec, model, cond):
    logger = logging.getLogger('results')
    pred_audio = np.array([])
    orig_mix_spec = normlize_complex(orig_mix_spec)
    orig_mix_mag = np.abs(orig_mix_spec)
    orig_mix_phase = np.angle(orig_mix_spec)
    pred_audio, pred_mag = None, None
    try:
        if config.MODE == 'standard':
            num_bands, num_frames = model.input_shape[1:3]
            x = prepare_a_song(orig_mix_mag, num_frames, num_bands)
            pred_mag = model.predict(x)
        if config.MODE == 'conditioned':
            num_bands, num_frames = model.input_shape[0][1:3]
            x, cond_seg = prepare_a_song(orig_mix_mag, num_frames, num_bands, cond)

            randname = random.randint(1,1000)
            #plot_f0(np.reshape(cond_seg,(-1,config_prepro.CQT_BINS+1)),str(randname))
            #plot_spec(np.squeeze(concatenate(x, orig_mix_spec.shape), axis=-1),orig_mix_phase,str(randname))
            # if config.EMB_TYPE == 'dense':
            #     cosnd = cond.reshape(1, -1)
            # if config.EMB_TYPE == 'cnn':
            #     cond = cond.reshape(-1, 1)
            # tmp = np.zeros((x.shape[0], *cond.shape))
            # tmp[:] = cond
            print('Prepared Spec X:'+str(np.shape(x)))
            print('Prepared Spec Cond:'+str(np.shape(cond_seg)))
            pred_mag = model.predict([x, cond_seg])

        pred_mag = np.squeeze(
            concatenate(pred_mag, orig_mix_spec.shape), axis=-1)
        pred_audio = reconstruct(
            pred_mag, orig_mix_phase, orig_mix_mag)
    except Exception as my_error:
        logger.error(my_error)
    return pred_audio, pred_mag


def do_an_exp(audio, target_source, model, file=''):
    accompaniment = np.zeros([1])
    for i in config.INSTRUMENTS:
        if i not in target_source:
            if accompaniment.size == 1:
                accompaniment = audio[i]
            else:
                accompaniment = np.sum([accompaniment, audio[i]], axis=0)

    # original isolate target
    target = istft(audio[target_source])
    # original mix
    mix = istft(audio['mixture'])
    # accompaniment (sum of all apart from the original)
    acc = istft(accompaniment)

    # predicted separation
    file_length = audio[target_source].shape[1]
    cond = np.zeros((file_length,config_prepro.CQT_BINS+1))
    cond = get_f0(file, target_source, '1') # Only cover use-case#1 here: exactly one singer per part

    print('Spec Target: '+str(np.shape(audio[target_source])))
    print('Spec Acc: '+str(np.shape(accompaniment)))
    print('Spec Mixture: '+str(np.shape(audio['mixture'])))

    print('Audio Target: '+str(np.shape(target)))
    print('Audio Acc: '+str(np.shape(acc)))
    print('Audio Mixture: '+str(np.shape(mix)))

    print('Cond: '+str(np.shape(cond)))
    pred_audio, pred_mag = analize_spec(audio['mixture'], model, cond)
    print('Pred Audio:'+str(np.shape(pred_audio)))
    print('Pred Mag:'+str(np.shape(pred_mag)))
    # to go back to the range of values of the original target
    pred_audio = adapt_pred(pred_audio, target)
    print('Pred Audio:'+str(np.shape(pred_audio)))
    # size
    s = min(pred_audio.shape[0], target.shape[0], mix.shape[0], acc.shape[0])
    pred_acc = mix[:s] - pred_audio[:s]
    pred = np.array([pred_audio[:s], pred_acc])
    orig = np.array([target[:s], acc[:s]])
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        reference_sources=target, estimated_sources=pred_audio,
        compute_permutation=False)
    save_pred_to_path(pred_audio,str((file+'_'+target_source+'_1')))
    return sdr[perm[0]], sir[perm[0]], sar[perm[0]]


def get_stats(dict, stat):
    logger = logging.getLogger('computing_spec')
    values = np.fromiter(dict.values(), dtype=float)
    r = {'mean': np.mean(values), 'std': np.std(values)}
    logger.info(stat + " : mean {}, std {}".format(r['mean'], r['std']))
    return r


def create_pandas(files):
    columns = ['name', 'target', 'sdr', 'sir', 'sar']
    if config.MODE == 'standard':
        data = np.zeros((len(files), len(columns)))
    else:
        data = np.zeros((len(files)*len(config.TARGET), len(columns)))
    df = pd.DataFrame(data, columns=columns)
    df['name'] = df['name'].astype('str')
    df['target'] = df['target'].astype('str')
    return df


def load_checkpoint(path_results, ckpt_path=''):
    from cunet.train.models.unet_model import unet_model
    from cunet.train.models.cunet_model import cunet_model

    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = cunet_model()

    path_results = os.path.join(path_results, 'checkpoint')

    if ckpt_path == '':
        latest = tf.train.latest_checkpoint(path_results)
        print('latest: '+str(latest))
        model.load_weights(latest)
    else:
        print('path_results: ' + str(ckpt_path))
        model.load_weights(os.path.join(path_results,ckpt_path))

    return model, path_results


def load_a_cunet(target=None):
    model = None
    if config.MODE == 'standard':
        path_results = os.path.join(config.PATH_MODEL, target, config.NAME)
        path_model = os.path.join(path_results, config.NAME+'.h5')
        if os.path.exists(path_model):
            model = load_model(path_model)
        else:
            model = load_checkpoint(path_results)
    else:
        path_results = os.path.join(config.PATH_MODEL, config.NAME)
        path_model = os.path.join(path_results, config.NAME+'.h5')
        if os.path.exists(path_model):
            model = load_model(path_model,  custom_objects={"tf": tf, "FiLM_simple_layer":FiLM_simple_layer})
        else:
            model = load_checkpoint(path_results)
    return model, path_results


def main():
    i = 0
    config.parse_args()
    files = glob(os.path.join(config.PATH_AUDIO, '*.npz'))
    if config.MODE == 'conditioned':
        results = create_pandas(files)
        model, path_results = load_checkpoint(os.path.join(config.PATH_MODEL,config.NAME),'ckpt_245-0.00409')
    for target in config.TARGET:
        if config.MODE == 'standard':
            i = 0
            results = create_pandas(files)
            model, path_results = load_a_cunet(target)
        file_handler = logging.FileHandler(
            os.path.join(path_results, 'results.log'),  mode='w')
        file_handler.setLevel(logging.INFO)
        logger = logging.getLogger('results')
        logger.addHandler(file_handler)
        logger.info('Starting the computation')
        for fl in files:
            print(fl)
            name = os.path.basename(os.path.normpath(fl)).replace('.npz', '')
            audio = np.load(fl, allow_pickle=True)
            logger.info('Song num: ' + str(i+1) + ' out of ' + str(len(results)))
            results.at[i, 'name'] = name
            results.at[i, 'target'] = target
            logger.info('Analyzing ' + name + ' for target ' + target)

            (results.at[i, 'sdr'], results.at[i, 'sir'],
             results.at[i, 'sar']) = do_an_exp(audio, target, model, file=name)
            logger.info(results.iloc[i])
            i += 1
        results.to_pickle(os.path.join(path_results, 'results.pkl'))
        logger.removeHandler(file_handler)
    return


if __name__ == '__main__':
    main()
