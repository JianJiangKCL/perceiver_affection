import warnings
warnings.filterwarnings("ignore")
from datasets.data_module import Modalities
from models.multi_modality_perceiver import MultiModalityPerceiver
from funcs.build_dataset import get_loader
from funcs.setup import parse_args, set_logger, set_trainer
from funcs.utils_funcs import load_state_dict_flexible_, set_seed
import os
import wandb
from trainers.BaselineTrainer import BaselineTrainer
from trainers.MultiTaskTrainer import MultiTaskTrainer
from trainers.IncrementTrainer import IncrementTrainer
import torch
import copy


def main(args):
	name_modalities = args.modalities
	file_prefix = '_'.join(name_modalities)
	root_dir = save_path = f"{args.results_dir}/{file_prefix}_lr{args.lr}_e{args.epochs}_seed{args.seed}_opt{args.optimizer}_" \
						   f"bs{args.batch_size}_beta{args.beta}"

	os.makedirs(save_path, exist_ok=True)


	modalities = [Modalities[name] for name in name_modalities]
	print(f"modalities: {modalities}")
	return 0
	sensitive_groups = ["gender", "age"]
	train_loader = get_loader(args, name_modalities, sensitive_groups, 'train_val')
	val_loader = get_loader(args, name_modalities, sensitive_groups, 'test')
	test_loader = get_loader(args, name_modalities, sensitive_groups, 'test')

	backbone = MultiModalityPerceiver(
		modalities=modalities,
		depth=args.depth,
		num_latents=args.num_latents,
		latent_dim=args.latent_dim,
		cross_heads=args.cross_heads,
		latent_heads=args.latent_heads,
		cross_dim_head=args.cross_dim_head,
		latent_dim_head=args.latent_dim_head,
		num_outputs=args.num_outputs,
		attn_dropout=0.,
		ff_dropout=0.,
		weight_tie_layers=True
	)

	if args.finetune:
		checkpoint = torch.load(args.finetune)
		backbone = load_state_dict_flexible_(backbone, checkpoint['state_dict'])

	if args.is_incremental:
		old_model = copy.deepcopy(backbone)
		model = IncrementTrainer(args, backbone, old_model, name_modalities, sensitive_groups)
	else:
		Trainer = BaselineTrainer if args.is_baseline else MultiTaskTrainer
		model = Trainer(args, backbone, name_modalities, sensitive_groups)

	logger = None
	if args.use_logger:
		logger = set_logger(args, root_dir)
	trainer = set_trainer(args, logger, save_path)
	if not args.test_only:
		trainer.fit(model, train_loader, val_loader)
		print('--------------finish training')
		trainer.save_checkpoint(f'{save_path}/checkpoint.pt')
	trainer.test(model, test_loader)
	wandb.finish()


if __name__ == "__main__":
	args = parse_args()
	# set random seed
	set_seed(args.seed)
	print(args)

	main(args)