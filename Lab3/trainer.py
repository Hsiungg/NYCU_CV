"""
Training Mask RCNN using detectron2
"""
import os
import argparse

# trainer
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.transforms import (
    ResizeShortestEdge,
    RandomRotation,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    RandomLighting,
)
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
# evaluator
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# build_detection_test_loader
from detectron2.data import build_detection_train_loader
# from detectron2.engine import hooks
# from detectron2.engine.hooks import BestCheckpointer

# TensorBoard
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import TensorboardXWriter
from detectron2.engine.hooks import PeriodicWriter

from dataloader import register_custom_dataset
# Set up argument parser
parser = argparse.ArgumentParser(
    description="Training Mask R-CNN with custom dataset")
parser.add_argument(
    '--output_dir',
    type=str,
    default="",
    help="The name of the directory which store data"
)
args = parser.parse_args()

# Set up Config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.INPUT.MASK_FORMAT = "bitmask"
# Set up directory path
DATASET_DIR = "data"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train")

# Register dataset
class_names = ["class1", "class2", "class3", "class4"]
DATASET_NAME = "my_instance_dataset"
register_custom_dataset(TRAIN_IMG_DIR, class_names, DATASET_NAME)
setup_logger()

cfg.DATASETS.TRAIN = (DATASET_NAME,)
# cfg.DATASETS.TEST = ("my_val",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6

# Pretrained model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Data Augmentation Configuration
augmentation_pipeline = [
    ResizeShortestEdge(short_edge_length=(
        640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice"),
    RandomRotation(angle=[-30, 30]),
    RandomBrightness(0.8, 1.2),      # ~similar to ColorJitter brightness
    RandomContrast(0.8, 1.2),        # ~similar to ColorJitter contrast
    RandomSaturation(0.8, 1.2),      # ~similar to ColorJitter saturation
    RandomLighting(0.7),             # ~adds color jittering based on PCA
]

cfg.INPUT.AUGMENTATIONS = [
    "ResizeShortestEdge",
    "RandomRotation",
    "RandomApply(ColorJitter)",
    "RandomBlur",
]


class MyMapper:
    """Apply data augmentation to mapper
    """
    # pylint: disable=redefined-outer-name

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.augmentations = augmentation_pipeline
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format)

        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image

        annos = [
            utils.transform_instance_annotations(
                annotation, transforms, image.shape[:2])
            for annotation in dataset_dict.pop("annotations")
        ]

        instances = utils.annotations_to_instances(annos, image.shape[:2])

        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32"))
        dataset_dict["instances"] = instances

        return dataset_dict


# Training Configuration
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.OPTIMIZER = "AdamW"
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.WEIGHT_DECAY = 0.0001
TARGET_EPOCH = 50
img_per_iter = cfg.SOLVER.IMS_PER_BATCH
DATASET_SIZE = 209
iter_per_epoch = DATASET_SIZE // img_per_iter

# Learning Rate Scheduler
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.MAX_ITER = TARGET_EPOCH * iter_per_epoch
cfg.SOLVER.STEPS = (4 * iter_per_epoch, 8 * iter_per_epoch)
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 2 * iter_per_epoch

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

# Loss
# cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"

# OUTPUT DIR
OUTPUT_ROOT = "./output"
# Change here to store in different directory
cfg.OUTPUT_DIR = os.path.join(OUTPUT_ROOT, args.output_dir)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# evaluator
# pylint: disable=pointless-string-statement
'''
cfg.TEST.EVAL_PERIOD = iter_per_epoch
evaluator = COCOEvaluator(
    "my_val",
    distributed=False,
    output_dir=cfg.OUTPUT_DIR,
    # use_fast_impl=True,
    tasks=["bbox"],
)
'''


class MyTrainer(DefaultTrainer):
    """ Set up trainer env
    """
    @classmethod
    def build_train_loader(cls, cfg):
        # pylint: disable=missing-kwoa
        return build_detection_train_loader(
            cfg,
            # mapper=MyMapper(cfg, is_train=True),
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        )
# val_loader = build_detection_test_loader(cfg, "my_val")


# trainer
trainer = MyTrainer(cfg)
trainer.register_hooks([
    PeriodicWriter([TensorboardXWriter("./logs")], period=iter_per_epoch),
])
OUT_YAML = os.path.join(cfg.OUTPUT_DIR, "output_config.yaml")
with open(OUT_YAML, "w", encoding='utf-8') as f:
    f.write(cfg.dump())

# start training
trainer.resume_or_load(resume=False)
trainer.train()
