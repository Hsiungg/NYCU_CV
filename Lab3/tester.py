import os
import json
import argparse
import torch
import numpy as np
from skimage.io import imread
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pycocotools import mask as mask_utils
from tqdm import tqdm


def add_custom_config(cfg):
    cfg.INPUT.AUGMENTATIONS = []
    cfg.SOLVER.OPTIMIZER = "AdamW"


def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Detectron2 Prediction and Save Output")
    parser.add_argument('--save_dir', type=str, default="mask_rcnn_R_50_FPN_3x",
                        help="Directory storing model and config")
    parser.add_argument('--model_name', type=str, default="model_final.pth",
                        help="Model weights filename")
    parser.add_argument('--image_dir', type=str, default="data/test_release",
                        help="Directory containing test images (.tif)")
    parser.add_argument('--output_json', type=str, default="test-results.json",
                        help="Output JSON filename")
    args = parser.parse_args()

    # Load file_name to image_id and size mapping
    image_id_json = os.path.join("data", "test_image_name_to_ids.json")
    with open(image_id_json, "r", encoding="utf-8") as f:
        image_infos = json.load(f)

    file_name_to_id = {info["file_name"]: info["id"] for info in image_infos}
    file_name_to_size = {info["file_name"]: (
        info["height"], info["width"]) for info in image_infos}

    # Set up config
    out_path = os.path.join("output", args.save_dir)
    out_yaml = os.path.join(out_path, "output_config.yaml")

    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(out_yaml)
    cfg.MODEL.WEIGHTS = os.path.join(out_path, args.model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    # Start inference
    results = []

    for file_name in tqdm(os.listdir(args.image_dir)):
        if not file_name.endswith(".tif"):
            continue

        if file_name not in file_name_to_id:
            continue

        image_path = os.path.join(args.image_dir, file_name)
        image = imread(image_path)
        if image is None:
            continue
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image[..., ::-1]

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        image_id = file_name_to_id[file_name]
        height, width = file_name_to_size[file_name]

        for i in range(len(instances)):
            mask = instances.pred_masks[i].numpy()

            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            box = instances.pred_boxes.tensor[i].tolist()
            x0, y0, x1, y1 = box
            bbox = [x0, y0, x1 - x0, y1 - y0]
            result = {
                "image_id": image_id,
                "bbox": bbox,
                "score": float(instances.scores[i]),
                "category_id": int(instances.pred_classes[i]) + 1,
                "segmentation": {
                    "size": [height, width],
                    "counts": rle["counts"]
                }
            }
            results.append(result)
    # Save results to JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Inference complete. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
