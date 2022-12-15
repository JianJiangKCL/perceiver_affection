from datasets.data_module import NpDataset, Modalities
from torch.utils.data import DataLoader
import os
from torchvision.datasets import ImageFolder


def get_loader(args, name_modalities, mode):
	dataset = NpDataset(os.path.join(args.dataset_path, f'{mode}_text_fb.npz'), name_modalities)
	if mode == 'train':
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
	elif mode == 'validation':
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

	else:
		raise ValueError(f'Invalid mode {mode}')

	return loader