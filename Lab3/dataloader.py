"""Register dataset to Detectron2 for Mask-RCNN"""
import os
import pickle
import numpy as np
from skimage.io import imread
from pycocotools import mask as mask_util
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def register_custom_dataset(root_dir, class_names, dataset_name="my_instance_dataset"):
    """
    transform dataset into dataset dicts for detectron2 API

    Args:
        root_dir (str): root directory that place train data
        class_names (List[str]): A list that list all the names of class
        dataset_name (str): Registered dataset name, default: "my_instance_dataset"

    """
    pkl_path = os.path.join(root_dir, f"{dataset_name}_dicts.pkl")

    if os.path.exists(pkl_path):
        print(f"Loading cached dataset from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            dataset_dicts = pickle.load(f)

    else:
        print("Building dataset from raw images and masks...")
        dataset_dicts = []

        for img_id, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)
            image_path = os.path.join(folder_path, "image.tif")
            if not os.path.exists(image_path):
                print(f"{image_path} not exist!")
                continue

            img = imread(image_path)
            height, width = img.shape[:2]

            record = {
                "file_name": image_path,
                "image_id": img_id,
                "height": height,
                "width": width,
                "annotations": []
            }
            for class_idx, class_name in enumerate(class_names):
                mask_path = os.path.join(folder_path, f"{class_name}.tif")
                if not os.path.exists(mask_path):
                    continue

                mask = imread(mask_path).copy()
                instance_ids = np.unique(mask)
                # remove background "0".
                instance_ids = instance_ids[instance_ids != 0]

                for inst_id in instance_ids:
                    binary_mask = (mask == inst_id).astype(
                        np.uint8, copy=True).copy()
                    if binary_mask.sum() == 0:
                        continue
                    binary_mask = np.asfortranarray(binary_mask.copy())
                    rle = mask_util.encode(binary_mask)
                    area = mask_util.area(rle).item()
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox = mask_util.toBbox(rle).tolist()

                    annotation = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": rle,
                        "category_id": class_idx,
                        "iscrowd": 0,
                        "area": area,
                    }
                    record["annotations"].append(annotation)

            dataset_dicts.append(record)
        with open(pkl_path, "wb") as f:
            pickle.dump(dataset_dicts, f)
        print(f"Saved cached dataset to: {pkl_path}")

    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
        MetadataCatalog.get(dataset_name).set(thing_classes=class_names)
        print(f"Registered dataset '{dataset_name}'")
    else:
        print(f"Dataset '{dataset_name}' is already registered.")
    print(
        f" Registered dataset '{dataset_name}' with {len(dataset_dicts)} samples.")
