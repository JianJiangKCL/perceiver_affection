from datasets.data_module import NpDataset, MultiTaskDataset

#######
## test npz dataset
# train_dataset = NpDataset('train_text_fb.npz', ['text', 'facebody'])
#
# val_dataset = NpDataset('validation_text_fb.npz', ['text', 'facebody'])
# dat = train_dataset[0]
#######


# test multitask dataset
train_dataset = MultiTaskDataset('train_text_fb.npz', ['text', 'facebody'], 'age')
test_dataset = MultiTaskDataset('test_text_fb.npz', ['text', 'facebody'], 'age')
data = train_dataset[0]
test_data = test_dataset[0]
k=1
