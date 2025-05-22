import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=200,
                    help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=2,  # Try batch size=4?
                    help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate of encoder.')
parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                    choices=['cosine_warmup', 'cosine_restart', 'one_cycle',
                             'reduce_on_plateau', 'multistep', 'none'],
                    help='Learning rate scheduler to use')

parser.add_argument('--de_type', nargs='+', default=['derain', 'desnow'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=256,
                    help='patchsize of input.')
parser.add_argument('--num_workers', type=int,
                    default=6, help='number of workers.')

# path
parser.add_argument('--training_root_dir', type=str, default='data/Train',
                    help='where training images saves.')
parser.add_argument('--output_path', type=str,
                    default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str,
                    default="", help='checkpoint save path')
parser.add_argument("--wblogger", type=str, default="promptir",
                    help="Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir", type=str, default="/mnt/sda1/cv/checkpoints/cut_256/",
                    help="Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs to use for training")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume training from")

options = parser.parse_args()
