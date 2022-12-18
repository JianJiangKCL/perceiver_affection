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

def convert_data(data_path, mode, save_path):
	print('build dataset...')
	labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']

	path = os.path.join(data_path, f'{mode}_data.csv')

	data = pd.read_csv(path)

	gender = data.gender.to_numpy()
	# map gender, M is 1, F is 0
	mapped_gender = np.array([1 if g == 'M' else 0 for g in gender])
	age = data.age.to_numpy()
	# map gender, <30 is 1, >=30 is 0
	mapped_age = np.array([1 if a < 30 else 0 for a in age])
	comb_OCEAN = get_OCEAN(data, labels).astype(np.float32)

	# following train data only has the feature data
	drop_cols = ['ID_y', 'minute', 'session', 'gender', 'age','Unnamed: 0', 'Video', 'Unnamed: 0.1']

	# combined
	comb = df_drop_cols(data, drop_cols)

	Fb_drop_cols = [(f'{i}_fb') for i in range(0, 552)]
	Bt_drop_cols = [(f'{i}_bt') for i in range(0, 512)]

	comb_bt = np.array(comb.drop([*Fb_drop_cols], axis=1)).astype(np.float32)
	comb_fb = np.array(comb.drop([*Bt_drop_cols], axis=1)).astype(np.float32)

	# gender = np.array
	np.savez(os.path.join(save_path, f'{mode}_text_fb.npz'), text=comb_bt, facebody=comb_fb, OCEAN=comb_OCEAN, age=mapped_age, gender=mapped_gender)


def combine_train_val(train_path, val_path, save_path):
	# combine each item in the dictionary, and save all of them to a new file
	train_data = np.load(train_path)
	val_data = np.load(val_path)
	new_data = {}
	for key in train_data.keys():
		train_tmp = train_data[key]
		val_tmp = val_data[key]
		combined_data = np.concatenate((train_tmp, val_tmp), axis=0)
		new_data[key] = combined_data
	np.savez(save_path, **new_data)


def main(args):
	data_path = args.data_path
	save_path = args.save_path
	# build npz files for train, val, and test
	# convert_data(data_path, 'train', save_path)
	#
	# convert_data(data_path, 'validation', save_path)
	# convert_data(data_path, 'test', save_path)

	data = np.load(os.path.join(save_path, 'train_text_fb.npz'))
	train_val_data = np.load(os.path.join(save_path, 'train_val_text_fb.npz'))
	age = data['age']
	train_val_age = train_val_data['age']
	k=1
	##################################
	# combine train and validation data
	# combine_train_val(os.path.join(save_path, 'train_text_fb.npz'), os.path.join(save_path, 'validation_text_fb.npz'), os.path.join(save_path, 'train_val_text_fb.npz'))

	##################################
	k=1




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", default='datasets/multi_modality_csv', type=str)
	parser.add_argument("--save_path", default='H:\project\perceiver_affection', type=str)
	parser.add_argument("--OCEAN_id", type=str)
	parser.add_argument("--setting", type=str)
	args = parser.parse_args()
	main(args)
