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
	# convert_data('multi_modality_csv', 'train')

	# convert_data('multi_modality_csv', 'validation')
	# convert_data('multi_modality_csv', 'test')
	data = np.load('train_text_fb.npz')

	##################################
	# combine train and validation data
	train_text = data['text']
	train_facebody = data['facebody']
	train_OCEAN = data['OCEAN']

	# data = np.load('validation_text_fb.npz')
	# # combine train and validation data
	# validation_text = data['text']
	# validation_facebody = data['facebody']
	# validation_OCEAN = data['OCEAN']
	#
	# train_text = np.concatenate((train_text, validation_text), axis=0)
	# train_facebody = np.concatenate((train_facebody, validation_facebody), axis=0)
	# train_OCEAN = np.concatenate((train_OCEAN, validation_OCEAN), axis=0)
	#
	# np.savez('train_text_fb.npz', text=train_text, facebody=train_facebody, OCEAN=train_OCEAN)
	##################################
	k=1




if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--OCEAN_id", type=str)
	parser.add_argument("--setting", type=str)
	args = parser.parse_args()
	main(args)
