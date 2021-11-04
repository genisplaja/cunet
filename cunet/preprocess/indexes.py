import numpy as np
import copy
import itertools
import os
from cunet.preprocess.config import config
from cunet.train.config import config as config_train
from glob import glob
import logging
import pandas as pd
from scipy import signal
from scipy.ndimage import filters
from random import randint
from matplotlib import pyplot as plt
from tqdm import tqdm
import librosa

DEBUG_DIR = './debug' 


def process_f0(f0, f_bins, n_freqs):
	freqz = np.zeros((f0.shape[0], f_bins.shape[0]))
	haha = np.digitize(f0, f_bins) - 1
	idx2 = haha < n_freqs
	haha = haha[idx2]
	freqz[range(len(haha)), haha] = 1
	atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T
	min_target = np.min(atb[range(len(haha)), haha])
	atb = atb / min_target
	atb[atb > 1] = 1
	return atb

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def condition2key(cond):
	key = np.array2string(cond).replace("[", "").replace("]", "")
	return ",".join([t for t in key.split(" ") if t != ''])

def frame2ms(frame):
	time_per_frame = np.float128(config.HOP * np.float128((1.0 / config.FR)))
	time_for_frame = np.float128(frame * time_per_frame)
	time_for_frame += config.FFT_SIZE

	return time_for_frame

def get_conditions():
	logger = logging.getLogger('getting_indexes')
	logger.info('Computing the conditions')
	conditions_raw = [
		np.array(i).astype(np.float)
		for i in list(itertools.product([0, 1], repeat=4))
		if np.sum(i) <= config.CONDITION_MIX and np.sum(i) > 0
	]
	conditions = []
	if config.ADD_ALL:      # add the all mix condition
		conditions_raw.append(np.ones(len(config.CONDITIONS)))
	keys = []
	in_between = np.arange(
		config.ADD_IN_BETWEEN, 1+config.ADD_IN_BETWEEN, config.ADD_IN_BETWEEN
	)
	for cond in conditions_raw:
		for index in np.nonzero(cond)[0]:
			# adding intermedia values to the conditions - in between idea
			for b in in_between:
				tmp = copy.deepcopy(cond)
				tmp[index] = tmp[index]*b
				key = condition2key(tmp)
				if key not in keys:     # avoiding duplicates
					conditions.append(tmp.astype(np.float32))
					keys.append(key)
	if config.ADD_ZERO:     # add the zero condition
		conditions.append(np.zeros(len(config.CONDITIONS)).astype(np.float32))
	logger.info('Done!')
	logger.info('Your conditions are %s', conditions)
	return conditions


def chunks(l, chunk_size):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), chunk_size):
		yield l[i:i + chunk_size]


# Get the time stamps from the specs. for F0s
def get_indexes():

	logger = logging.getLogger('getting_indexes')
	logger.info('Computing the indexes')
	indexes = {'config': {'FR': config.FR, 'FFT_SIZE': config.FFT_SIZE, 'HOP': config.HOP}}

	try:

		# Get all the spec_files. from NPZ (for each songs SONG[GROUP][NUMBER])
		spec_files = glob(os.path.join(config.PATH_SPEC, '*.npz'))

		# Get all the f0s files
		f0s_files  = glob(os.path.join(config.PATH_F0S, '*.csv'),recursive=False)
		print(f0s_files)

		# Iterate through all previously computed specs. 
		for f in tqdm(np.random.choice(spec_files, len(spec_files), replace=False)):
			print(f)

			logger.info('Input points for track %s' % f)

			# SATB
			spec = np.load(f,allow_pickle=True)
			name = os.path.basename(os.path.normpath(f)).replace('.npz', '') # stem name saved along indexes
			indexes[name] = dict()

			# Iterate through all groups
			for s_group in spec.files:
				print('s_group', s_group)

				if not any(x in s_group for x in ['config','mixture']):
					print('hola')

					indexes[name][s_group] = dict()

					# Iterate through all parts
					for i in spec[s_group].item().keys():
						print('adeu')
						s = []

						indexes[name][s_group][str(i)] = None
						# Get the number of frames for current spec.
						file_length = spec[s_group].item()[str(i)].shape[1]
						# Retrieve F0s file for current spec
						part_name = str(name+'_'+s_group+'_'+i)
						f0_file = [s for s in f0s_files if part_name in s]
						f0_data = pd.read_csv(f0_file[0],names=["frame", "f0"]) 
						f0_frame = f0_data['f0']
						f0_resampled = signal.resample(f0_frame,file_length)

						# One-hot encode F0 track
						freq_grid = librosa.cqt_frequencies(config.CQT_BINS,config.MIN_FREQ,config.BIN_PER_OCT)
						f_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])
						n_freqs = len(freq_grid)

						atb = process_f0(f0_resampled, f_bins, n_freqs)

						# for j in np.arange(0, file_length, config.STEP): # iterate over all spec frames

						# 	s.append([j, atb[j,:]])

						# s = np.asarray(atb, dtype=float)
						logger.info('indexes computed for group %s, part %s with shape %s' % (s_group,i,str(np.shape(atb))))
						indexes[name][s_group][str(i)] = atb

	except Exception as error:
		logger.error(error)
	return indexes


def main():
	logging.basicConfig(
		filename=os.path.join(config.PATH_INDEXES, 'getting_indexes.log'),
		level=logging.INFO
	)
	logger = logging.getLogger('getting_indexes')
	logger.info('Starting the computation')
	conditions = []
	name = "_".join(['indexes','SSSS','F0s'])
	data = get_indexes()
	logger.info('Saving')
	np.savez(
        os.path.join(config.PATH_INDEXES, name+'.npz'),**data)
	logger.info('Done!')
	return


if __name__ == "__main__":
	config.parse_args()
	main()
