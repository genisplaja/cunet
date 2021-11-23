from ftanet.evaluator import get_pitch_track
from preprocess import spectrogram, indexes
from evaluation import results

input_audio = '/home/genis/cunet/resources/Saraga-example-tracks/Thiruveragane_Saveri_Varnam_165.wav'
_, f0 = get_pitch_track(input_audio)

spec_data = spectrogram.compute_one_file(input_audio)
index_data = indexes.get_index_for_file(f0, spec_data['spec'])

model, _ = results.load_a_cunet()
results.single_separation(spec_data['spec'], index_data['data'], model, file='prova')