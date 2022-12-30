import warnings
warnings.filterwarnings("ignore")
from datasets.data_module import Modalities
from models.multi_modality_perceiver import MultiModalityPerceiver
from funcs.build_dataset import get_loader
from funcs.setup import parse_args, set_logger, set_trainer
from funcs.utils_funcs import load_state_from_ddp, set_seed
import os
import wandb
from trainers.BaselineTrainer import BaselineTrainer
from trainers.MultiTaskTrainer import MultiTaskTrainer


def main(args):

	root_dir = f"{args.results_dir}/{args.arch}_lr{args.lr}_e{args.epochs}_seed{args.seed}"
	save_path = os.path.join(root_dir, f'task{args.task_id}')
	os.makedirs(save_path, exist_ok=True)

	name_modalities = ['text', 'facebody']
	modalities = [Modalities[name] for name in name_modalities]

	train_loader = get_loader(args, name_modalities, 'train_val')
	val_loader = get_loader(args, name_modalities, 'test')
	test_loader = get_loader(args, name_modalities, 'test')

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
	Trainer = MultiTaskTrainer if args.multi_task else BaselineTrainer
	model = Trainer(args, backbone, name_modalities)

	logger = None
	if args.use_logger:
		logger = set_logger(args, root_dir)
	trainer = set_trainer(args, logger, save_path)
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