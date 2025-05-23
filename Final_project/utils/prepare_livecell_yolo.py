import os
import json
import numpy as np
import shutil
import cv2
from pycocotools.coco import COCO
import argparse
from tqdm import tqdm
from skimage import io, exposure
from collections import defaultdict


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_coco_annotations(annotation_path):
    print(f"Loading annotations from: {annotation_path}")
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            print(f"JSON file loaded successfully. Keys: {data.keys()}")

            # Create a new COCO format dictionary
            coco_format = {
                'images': data['images'],
                'annotations': [],
                'categories': data['categories'],
                'info': data['info'],
                'licenses': data['licenses']
            }

            # Convert annotations from dict to list
            if isinstance(data['annotations'], dict):
                annotations_list = list(data['annotations'].values())
                annotations_list.sort(key=lambda x: x['image_id'])
                coco_format['annotations'] = annotations_list
            else:
                coco_format['annotations'] = data['annotations']

            # Save the converted format
            temp_path = annotation_path + '.temp'
            with open(temp_path, 'w') as f:
                json.dump(coco_format, f)

            # Load the converted format
            coco = COCO(temp_path)

            # Clean up temporary file
            os.remove(temp_path)

            return coco

    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        raise


def get_cell_type_from_filename(filename):
    return filename.split('_')[0]


def convert_to_yolo_format(coco_annotation_path, base_image_dir, output_dir, cell_type=None, enhance=True):
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    create_directory(images_dir)
    create_directory(labels_dir)

    # Create train and val subdirectories
    train_img_dir = os.path.join(images_dir, 'train')
    train_label_dir = os.path.join(labels_dir, 'train')
    val_img_dir = os.path.join(images_dir, 'val')
    val_label_dir = os.path.join(labels_dir, 'val')

    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        create_directory(dir_path)

    # Load COCO annotations
    print("Loading annotations...")
    coco = load_coco_annotations(coco_annotation_path)

    # Get all image IDs
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images")

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if ann_ids:  # Only add if there are annotations
            annotations_by_image[img_id] = coco.loadAnns(ann_ids)

    # Process each image with its annotations
    print("Processing images and annotations...")
    for img_id, anns in tqdm(annotations_by_image.items(), desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']

        # Get cell type from filename
        cell_type_from_file = get_cell_type_from_filename(img_filename)

        # Skip if cell type doesn't match
        if cell_type is not None and cell_type_from_file.lower() != cell_type.lower():
            continue

        # Read original image
        img_path = os.path.join(
            base_image_dir, cell_type_from_file, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            continue

        # Determine if this is a training or validation image
        is_train = 'train' in coco_annotation_path.lower()
        target_img_dir = train_img_dir if is_train else val_img_dir
        target_label_dir = train_label_dir if is_train else val_label_dir

        # Read image using skimage
        img = io.imread(img_path)

        # Convert to RGB if needed
        if len(img.shape) == 2:  # If grayscale
            # Convert to RGB by repeating the channel
            img = np.stack([img] * 3, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 1:  # If single channel
            # Convert to RGB by repeating the channel
            img = np.repeat(img, 3, axis=2)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # If RGBA
            img = img[:, :, :3]  # Remove alpha channel

        # Enhance contrast if enabled
        if enhance:
            enhanced_img = np.zeros_like(img)
            for c in range(3):
                # Apply contrast stretching
                p2, p98 = np.percentile(img[:, :, c], (2, 98))
                enhanced_img[:, :, c] = exposure.rescale_intensity(
                    img[:, :, c], in_range=(p2, p98))
            output_img = enhanced_img
        else:
            output_img = img

        # Save image in RGB format
        output_img_path = os.path.join(target_img_dir, img_filename)
        io.imsave(output_img_path, output_img)

        # Create YOLO format label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(target_label_dir, label_filename)

        # Get image dimensions
        img_height, img_width = output_img.shape[:2]

        # Process annotations and remove duplicates
        unique_boxes = []
        for ann in anns:
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height]

            # Convert to YOLO format (x_center, y_center, width, height) normalized
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height

            # Create box tuple
            # Assuming class_id is 0
            box = (0, x_center, y_center, width, height)
            unique_boxes.append(box)

        # Write unique boxes to file
        with open(label_path, 'w') as f:
            for box in unique_boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LiveCell dataset to YOLO format')
    parser.add_argument('--dataset_type', type=str, choices=['full', 'single_cell'], default='full',
                        help='Choose between full LIVECell dataset or single cell type dataset')
    parser.add_argument('--cell_type', type=str, default=None,
                        help='Cell type to process (e.g., a172, bt474, etc.)')
    parser.add_argument('--enhance', action='store_true', default=False,
                        help='Enable image enhancement (contrast stretching)')
    args = parser.parse_args()

    # Define paths
    base_dir = 'data/LIVECell_dataset_2021'
    output_dir = 'data/livecell_yolo'

    if args.dataset_type == 'single_cell':
        if args.cell_type is None:
            raise ValueError(
                "cell_type must be specified when using single_cell dataset type")
        cell_type = args.cell_type.lower()
        # Create cell-type specific output directory
        output_dir = os.path.join(output_dir, cell_type)
        train_anno_path = os.path.join(
            base_dir, 'annotations', 'LIVECell_single_cells', cell_type, f'livecell_{cell_type}_train.json')
        val_anno_path = os.path.join(
            base_dir, 'annotations', 'LIVECell_single_cells', cell_type, f'livecell_{cell_type}_val.json')
        filter_cell_type = cell_type
    else:
        train_anno_path = os.path.join(
            base_dir, 'annotations', 'LIVECell', 'livecell_coco_train.json')
        val_anno_path = os.path.join(
            base_dir, 'annotations', 'LIVECell', 'livecell_coco_val.json')
        filter_cell_type = None

    base_image_dir = os.path.join(
        base_dir, 'images', 'livecell_train_val_images')

    print("\nProcessing training set...")
    convert_to_yolo_format(train_anno_path, base_image_dir,
                           output_dir, filter_cell_type, enhance=args.enhance)

    print("\nProcessing validation set...")
    convert_to_yolo_format(val_anno_path, base_image_dir,
                           output_dir, filter_cell_type, enhance=args.enhance)

    # Create dataset.yaml file
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: 1
names: ['cell']
"""

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(
        f"\nDataset conversion complete. YOLO format dataset saved to: {output_dir}")


if __name__ == '__main__':
    main()
