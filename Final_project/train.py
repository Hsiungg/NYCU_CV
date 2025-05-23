import os
from cellpose import models, core, io, plot, train
from pathlib import Path
from tqdm import trange
from custom_train import train_seg_with_callback 
from torch.utils.tensorboard import SummaryWriter
import torch
from check_bad_patch import filter_invalid_patches

# --------------------------- set up ----------------------------- #
io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")

# ------------------------- training params ------------------------- #
model_name = "slice_epoch100_bs2"
save_path = Path("checkpoints")
if not save_path.exists():
  save_path.mkdir(parents=True, exist_ok=True)

# default training params
n_epochs = 100
learning_rate = 1e-5
weight_decay = 0.1
batch_size = 2

train_dir = "sartorius-cell-instance-segmentation/train_tiny/"
if not Path(train_dir).exists():
  raise FileNotFoundError("directory does not exist")
test_dir = "sartorius-cell-instance-segmentation/val_tiny/"
masks_ext = "_seg.npy"

# list all files
files = [f for f in Path(train_dir).glob("*") if "_masks" not in f.name and "_flows" not in f.name and "_seg" not in f.name]
if(len(files)==0):
  raise FileNotFoundError("no files found, did you specify the correct folder and extension?")
else:
  print(f"{len(files)} files in folder:")

# --------------------------- set up tensorboard ----------------------------- #
log_dir = os.path.join("runs", model_name)
# create folder if not exists
os.makedirs(log_dir, exist_ok=True)

# create tensorboard writer
writer = SummaryWriter(log_dir)

# --------------------------- train model ----------------------------- #
model = models.CellposeModel(gpu=True)
# get files
output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)
train_data, train_labels, _, test_data, test_labels, _ = output

mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 檢查訓練集
train_data, train_labels, bad_train_indices = filter_invalid_patches(
    train_data, train_labels, prefix='train', device=mydevice
)

# 檢查驗證集
test_data, test_labels, bad_test_indices = filter_invalid_patches(
    test_data, test_labels, prefix='val', device=mydevice
)


print('[Info] All defective patches have been removed.')
print('[Info] Training and validation sets are now clean and ready for robust Cellpose training.')

new_model_path, train_losses, test_losses = train_seg_with_callback(model.net,
                                                            train_data=train_data,
                                                            train_labels=train_labels,
                                                            test_data=test_data,
                                                            test_labels=test_labels,
                                                            batch_size=batch_size,
                                                            n_epochs=n_epochs,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            nimg_per_epoch=max(2, len(train_data)), # can change this
                                                            save_every=5,
                                                            save_path=save_path,
                                                            save_each = True,
                                                            model_name=model_name,
                                                            tb_writer=writer,
                                                            validate_every=1,
                                                            min_train_masks=0,
                                                            normalize=False,)
print(f"model saved to {new_model_path}")
# close tensorboard
writer.close()