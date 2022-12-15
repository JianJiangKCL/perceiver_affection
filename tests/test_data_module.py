from datasets.data_module import NpDataset

train_dataset = NpDataset('train_text_fb.npz', ['text', 'facebody'])

val_dataset = NpDataset('validation_text_fb.npz', ['text', 'facebody'])
dat = train_dataset[0]
k=1
