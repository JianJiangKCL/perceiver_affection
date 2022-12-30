from datasets.data_module import NpDataset, Modalities, MultiTaskDataset
from torch.utils.data import DataLoader
import os
from torchvision.datasets import ImageFolder


def get_loader(args, name_modalities, mode):
	dataset_path = os.path.join(args.dataset_path, f'{mode}_text_fb.npz')
	if args.multi_task:
		assert args.sensitive_group is not None
		dataset = MultiTaskDataset(dataset_path, name_modalities, sensitive_group=args.sensitive_group)
	else:
		dataset = NpDataset(dataset_path, name_modalities)
	if mode == 'train' or mode == 'train_val':
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
	elif mode == 'validation' or mode == 'test':
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

	else:
		raise ValueError(f'Invalid mode {mode}')

	return loader