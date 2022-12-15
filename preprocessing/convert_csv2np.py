#%%

import os
import h5py
import numpy as np
import pickle

import pandas as pd
import argparse
#%%
def df_drop_cols(df, cols):
	for col in cols:
		try:
			df.drop(
				[col], axis=1, inplace=True)
		except:
			pass
	return df



# these are ground truth for each OCEAN; we will use them to evaluate the performance of the model.
	# [1,N]
def get_OCEAN(d, labels):
	OCEAN = []
	for i in range(0, 5):
		OCEAN.append(d.pop(labels[i]).to_numpy())
	OCEAN = np.array(OCEAN)
	# permute the axis
	OCEAN = np.transpose(OCEAN)
	return OCEAN

def convert_data(root, mode):
	print('build dataset...')
	labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']

	path = os.path.join(root, f'{mode}_data.csv')

	data = pd.read_csv(path)

	comb_OCEAN = get_OCEAN(data, labels).astype(np.float32)

	# following train data only has the feature data
	drop_cols = ['ID_y', 'minute', 'session', 'gender', 'age','Unnamed: 0', 'Video', 'Unnamed: 0.1']

	# combined
	comb = df_drop_cols(data, drop_cols)

	Fb_drop_cols = [(f'{i}_fb') for i in range(0, 552)]
	Bt_drop_cols = [(f'{i}_bt') for i in range(0, 512)]

	comb_bt = np.array(comb.drop([*Fb_drop_cols], axis=1)).astype(np.float32)
	comb_fb = np.array(comb.drop([*Bt_drop_cols], axis=1)).astype(np.float32)

	np.savez(f'{mode}_text_fb.npz', text=comb_bt, facebody=comb_fb, OCEAN=comb_OCEAN)


def main(args):
	convert_data('multi_modality_csv', 'train')

	convert_data('multi_modality_csv', 'validation')
	data = np.load('train_text_fb.npz')
	text = data['text']
	facebody = data['facebody']
	OCEAN = data['OCEAN']
	k=1
	#
	# comb_train_set, Y_train_set, O_train_set = convert_data('multi_modality_csv', 'train', 'age')
	#
	# comb_val_set, Y_val_set, O_val_set = convert_data('multi_modality_csv', 'validation', 'age')

	# comb_test_set , M_test_set, F_test_set = convert_data('multi_modality_csv', 'test')



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--OCEAN_id", type=str)
	parser.add_argument("--setting", type=str)
	args = parser.parse_args()
	main(args)
