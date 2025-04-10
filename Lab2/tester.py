import os
import json
import csv
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

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

# Set up config
OUT_ROOT = "output"
# Change here to get yaml file
OUT_PATH = os.path.join(OUT_ROOT, args.output_dir)
OUT_YAML = os.path.join(OUT_PATH, "output_config.yaml")

cfg = get_cfg()
cfg.merge_from_file(OUT_YAML)
cfg.MODEL.WEIGHTS = os.path.join(OUT_PATH, "best_model.pth")

# set hyperparameters
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # confidence score
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
# Predictor and dataset
predictor = DefaultPredictor(cfg)
test_image_dir = 'nycu-hw2-data/test'
test_images = [f for f in os.listdir(test_image_dir) if f.endswith('.png')]
test_images = sorted(test_images, key=lambda f: int(f.split('.')[0]))
# Output in COCO format
coco_output = []
csv_results = []
# Process each image
for image_name in tqdm(test_images, desc="Testing...", unit="image"):
    image_path = os.path.join(test_image_dir, image_name)
    image = cv2.imread(image_path)

    # Predict image
    outputs = predictor(image)

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    # Sort by x-axis
    sorted_indices = np.argsort(boxes[:, 0])
    sorted_boxes = boxes[sorted_indices]
    sorted_scores = scores[sorted_indices]
    sorted_classes = classes[sorted_indices]

    for idx, bbox in enumerate(sorted_boxes):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        coco_annotation = {
            "image_id": int(image_name.split('.')[0]),
            "bbox": [float(xmin), float(ymin), float(width), float(height)],
            "score": float(sorted_scores[idx]),
            "category_id": int(sorted_classes[idx]) + 1
        }
        coco_output.append(coco_annotation)

    pred_label = ''.join([str(int(sorted_classes[i]))
                          for i in range(len(sorted_classes))])
    image_id = int(image_name.split('.')[0])
    csv_results.append({"image_id": image_id, "pred_label": pred_label})

json_output_path = os.path.join(OUT_PATH, 'pred.json')
with open(json_output_path, "w") as f:
    json.dump(coco_output, f, indent=4)

csv_output_path = os.path.join(OUT_PATH, 'pred.csv')
with open(csv_output_path, mode='w', newline='') as csvfile:
    fieldnames = ['image_id', 'pred_label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in csv_results:
        if not result.get('pred_label'):
            result['pred_label'] = -1
        writer.writerow(result)
    csvfile.flush()
