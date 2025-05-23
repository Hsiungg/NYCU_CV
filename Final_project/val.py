from cellpose import models, core, io, metrics

# --------------------- set up ----------------------- #
io.logger_setup() # run this to get printing of progress
# Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")

# ------------------------ params ------------------------ #
checkpoints_dir = "checkpoints/split1_model_epoch100_bs2/split1_model_epoch100_bs2"
test_dir = "sartorius-cell-instance-segmentation/myval1/" 
masks_ext = "_seg.npy"

# -------------------------- load data ------------------------- #
output = io.load_train_test_data(test_dir, mask_filter=masks_ext)
test_data, test_labels, _, _, _, _ = output

# -------------------- load model and validate ----------------------- #
# run model on test images and get masks
model = models.CellposeModel(gpu=True,
                             pretrained_model=checkpoints_dir)
masks = model.eval(test_data, batch_size=32)[0]


# ----------------------- calculate score -------------------------- #
# check performance using ground truth labels
ap = metrics.average_precision(test_labels, masks, threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])[0]
print('>>> average precision at different iou thresholds:')
for i, th in enumerate([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    print(f'>>> average precision at iou threshold {th}= {ap[:,i].mean():.3f}')

print('')
# 每張圖的 AP：mean over 10 thresholds
per_image_ap = ap.mean(axis=1)  # shape = (num_images,)

# 最後比賽評分分數：mean over all images
final_score = per_image_ap.mean()
print(f'>>> final score (mean average precision over all images) = {final_score:.4f}')