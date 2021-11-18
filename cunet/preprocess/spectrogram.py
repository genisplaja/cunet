import numpy as np
from glob import glob
from tqdm import tqdm
from cunet.preprocess.config import config
import logging
import os
import librosa


def get_config_as_str():
    return {
        'FR': config.FR, 'FFT_SIZE': config.FFT_SIZE,
        'HOP': config.HOP
    }


def spec_complex(audio_file):
    """Compute the complex spectrum"""
    output = {'type': 'complex'}
    logger = logging.getLogger('computing_spec')
    try:
        audio = np.zeros([1])
        logger.info('Computing complex spec for %s' % audio_file)
        audio, fe = librosa.load(audio_file, sr=config.FR)
        output['spec'] = librosa.stft(
            audio, n_fft=config.FFT_SIZE, hop_length=config.HOP)
    except Exception as my_error:
        logger.error(my_error)
    return output


def spec_mag(audio_file, norm=True):
    """Compute the normalized mag spec and the phase of an audio file"""
    output = {}
    logger = logging.getLogger('computing_spec')
    try:
        spec = spec_complex(audio_file)
        spec = spec['spec']
        logger.info('Computing mag and phase for %s' % audio_file)
        # n_freq_bins -> connected with fft_size with 1024 -> 513 bins
        # the number of band is odd -> removing the last band
        n = spec.shape[0] - 1
        mag = np.abs(spec[:n, :])
        #  mag = mag / np.max(mag)
        if norm:
            mx = np.max(mag)
            mn = np.min(mag)
            #  betweens 0 and 1 (x - min(x)) / (max(x) - min(x))
            mag = ((mag - mn) / (mx-mn))
            output['norm_param'] = np.array([mx, mn])
        output['phase'] = np.angle(spec)
        output['magnitude'] = mag
    except Exception as my_error:
        logger.error(my_error)
    return output


def spec_mag_log(audio_file):
    """Compute the normalized log mag spec and the phase of an audio file"""
    output = {}
    logger = logging.getLogger('computing_spec')
    try:
        spec = spec_mag(audio_file, False)    # mag without norm
        mag = spec['magnitude']
        output['phase'] = spec['phase']
        spec_log = np.log1p(mag)
        mx = np.max(spec_log)
        mn = np.min(spec_log)
        output['norm_param'] = np.array([mx, mn])
        output['log_magnitude'] = (spec_log - mn) / (mx - mn)
    except Exception as my_error:
        logger.error(my_error)
    return output


def compute_one_song(folder):
    logger = logging.getLogger('computing_spec')
    name = folder.split('/')[-2].replace('.wav', '')

    logger.info('Computing spec for %s' % name)

    # # MUSDB
    # for i in config.INTRUMENTS:
    #     print(folder+i+'.wav')
    # data = {i: spec_complex(folder+i+'.wav')['spec'] for i in config.INTRUMENTS}
    
    # SATB
    # count = 0
    # data = {i: dict() for i in config.CONDITIONS}
    # stem_list = glob(os.path.join(folder, "*.wav"))

    # SSSS
    count = 0
    data = {i: dict() for i in config.INSTRUMENTS}
    stem_list = list(glob(os.path.join(folder, "*mix.wav")))
    if stem_list:
        for i in stem_list:
            count += 1
            filename = i.split('/')[-1]
            folder_name = i.split('/')[-2]
            print('processing '+str(count)+' of '+str(len(stem_list))+' files')

            song = folder_name
            group = filename.split('_')[-1].replace('.wav', '')
            part = filename.split('_')[-2]

            if config.GROUP == 'test':
                data[group] = spec_complex(i)['spec']
                #data[group] = spec_complex(mix_stem)['spec']
            if config.GROUP == 'train':
                data['vocals'][part] = spec_complex(i)['spec']
                data['mixture'][part] = spec_complex(i.replace('mix', 'vocals'))['spec']

        if config.GROUP == 'test':
            data['mixture'] = spec_complex(stem_list)['spec']

        print('Saving data...')
        print(os.path.join(config.PATH_SPEC, name+'.npz'))
        np.savez(
            os.path.join(config.PATH_SPEC, name+'.npz'),
            config=get_config_as_str(), **data
        )
        return
    else:
        print('Empty folder!!')
        return


def main():
    logging.basicConfig(
        filename=os.path.join(config.PATH_SPEC, 'computing_spec.log'),
        level=logging.INFO
    )
    logger = logging.getLogger('computing_spec')
    logger.info('Starting the computation')
    for i in tqdm(glob(os.path.join(config.PATH_RAW, '*/'))):
        print(i)
        if os.path.isdir(i):
            compute_one_song(i)
    return


if __name__ == '__main__': 
    config.parse_args()
    main()
