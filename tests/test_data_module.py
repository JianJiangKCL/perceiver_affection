import numpy as np
import os
# comb_bt = np.array([1])
# comb_fb = np.array([2])
# comb_senti = np.array([3])
# comb_speech = np.array([4])
# total_modality_name = ['text', 'facebody', 'senti', 'speech']
# total_modality = [comb_bt, comb_fb, comb_senti, comb_speech]
#
# # gender = np.array
# # the file name is combined strings in the list
# file_name = '_'.join(total_modality_name)
#
# np.savez(os.path.join('save_path',  f'{file_name}.npz'), **dict(zip(total_modality_name, total_modality)))
values = 'text'
# values = ['text', 'face', '3', '4']
if isinstance(values, str):
	new_values = values.split(',')
else:
	new_values = values
k=1