"""
Training Faster RCNN using detectron2
"""
import os
import argparse

# trainer
from detectron2.data.datasets import register_coco_instances
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
import torch

# evaluator
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import hooks
from detectron2.engine.hooks import BestCheckpointer

# TensorBoard
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import TensorboardXWriter
from detectron2.engine.hooks import PeriodicWriter
# Set up directory path
DATASET_DIR = "nycu-hw2-data"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "valid")
TRAIN_ANN = os.path.join(DATASET_DIR, "train.json")
VAL_ANN = os.path.join(DATASET_DIR, "valid.json")

# Register dataset with COCO dataset format
register_coco_instances("my_train", {}, TRAIN_ANN, TRAIN_IMG_DIR)
register_coco_instances("my_val", {}, VAL_ANN, VAL_IMG_DIR)
setup_logger()

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Detectron2 Prediction and Save Output")
parser.add_argument(
    '--output_dir',
    type=str,
    default="faster_rcnn",
    help="The name of the directory which store data"
)
args = parser.parse_args()

# Set up Config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
))

cfg.DATASETS.TRAIN = ("my_train",)
cfg.DATASETS.TEST = ("my_val",)
cfg.DATALOADER.NUM_WORKERS = 6

# Pretrained model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

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
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.OPTIMIZER = "AdamW"
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.WEIGHT_DECAY = 0.0001
target_epoch = 20
img_per_iter = cfg.SOLVER.IMS_PER_BATCH
dataset_size = 30062
iter_per_epoch = dataset_size // img_per_iter
cfg.SOLVER.MAX_ITER = target_epoch * iter_per_epoch
cfg.SOLVER.STEPS = (10 * iter_per_epoch, 20 * iter_per_epoch)
cfg.SOLVER.GAMMA = 0.1

# Learning Rate Scheduler
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.MAX_ITER = target_epoch * iter_per_epoch
cfg.SOLVER.STEPS = (10 * iter_per_epoch, 20 * iter_per_epoch)
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 1.5 * iter_per_epoch

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10


# OUTPUT DIR
OUTPUT_ROOT = "./output"
# Change here to store in different directory
cfg.OUTPUT_DIR = os.path.join(OUTPUT_ROOT, args.output_dir)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# evaluator
cfg.TEST.EVAL_PERIOD = iter_per_epoch
evaluator = COCOEvaluator(
    "my_val",
    distributed=False,
    output_dir=cfg.OUTPUT_DIR,
    # use_fast_impl=True,
    tasks=["bbox"],
)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Instead of passing mapper= here directly...
        return build_detection_train_loader(
            cfg,
            mapper=MyMapper(cfg, is_train=True)
        )


val_loader = build_detection_test_loader(cfg, "my_val")

# trainer
trainer = MyTrainer(cfg)
trainer.register_hooks([
    PeriodicWriter([TensorboardXWriter("./logs")], period=iter_per_epoch),
    hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: inference_on_dataset(
        trainer.model, val_loader, evaluator)),
    BestCheckpointer(
        cfg.TEST.EVAL_PERIOD,
        trainer.checkpointer,
        "bbox/AP",
        mode="max",
        file_prefix="best_model"
    )
])
OUT_YAML = os.path.join(cfg.OUTPUT_DIR, "output_config.yaml")
with open(OUT_YAML, "w") as f:
    f.write(cfg.dump())

# start training
trainer.resume_or_load(resume=False)
trainer.train()
