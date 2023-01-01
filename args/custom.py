from argparse import ArgumentParser


def custom_args(parser: ArgumentParser):
	parser.add_argument("--n_emb", type=int, help=' the size of codebook, i.e. the number of embeddings', default=512)
	parser.add_argument("--use_qtz_only", choices=[0, 1], type=int, help='whether to random initial temporary weights', default=0)
	parser.add_argument("--beta", default=0.5, type=float)
	parser.add_argument("--gs", default=1, type=int, help="group size")
	parser.add_argument("--use_recon_codes", choices=[0, 1], type=int, help='whether to use previous reconstructed codes', default=1)
	parser.add_argument("--task_id", default=0, type=int)
	parser.add_argument("--pretrained_end_class", default=1000, type=int)
	parser.add_argument("--ckpt_root", default='', type=str)

	parser.add_argument("--depth", default=4, type=int)
	parser.add_argument("--num_latents", default=16, type=int)
	parser.add_argument("--latent_dim", default=512, type=int)
	parser.add_argument("--cross_heads", default=1, type=int)
	parser.add_argument("--latent_heads", default=8, type=int)
	parser.add_argument("--cross_dim_head", default=64, type=int)
	parser.add_argument("--latent_dim_head", default=64, type=int)
	parser.add_argument("--num_outputs", default=5, type=int)
	parser.add_argument("--multi_task", choices=[0, 1], type=int, default=1)
	parser.add_argument("--sensitive_group", default='gender', type=str)
	parser.add_argument("--is_baseline", choices=[0, 1], type=int, default=0)

