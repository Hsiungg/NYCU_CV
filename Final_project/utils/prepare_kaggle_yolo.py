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
            return data
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        raise


def convert_to_yolo_format(annotation_path, image_dir, output_dir, is_train=True, enhance=True):
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
    coco_data = load_coco_annotations(annotation_path)

    # Create image_id to filename mapping
    image_id_to_filename = {img['id']: img['file_name']
                            for img in coco_data['images']}

    # Create category mapping
    category_id_to_name = {cat['id']: cat['name']
                           for cat in coco_data['categories']}

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_id_to_filename:
            annotations_by_image[image_id].append(ann)

    # Process each image with its annotations
    print("Processing images and annotations...")
    for image_id, annotations in tqdm(annotations_by_image.items(), desc="Processing images"):
        img_filename = image_id_to_filename[image_id]
        img_filename = img_filename.split('/')[-1]
        img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            continue

        # Determine target directories
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

        # Write all annotations for this image
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get bounding box
                bbox = ann['bbox']  # [x, y, width, height]

                # Convert to YOLO format (x_center, y_center, width, height) normalized
                x_center = (bbox[0] + bbox[2]/2) / img_width
                y_center = (bbox[1] + bbox[3]/2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height

                # Get category ID (subtract 1 to make it 0-based for YOLO)
                category_id = ann['category_id'] - 1

                # Write to file
                f.write(
                    f"{category_id} {x_center} {y_center} {width} {height}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Kaggle COCO dataset to YOLO format')
    parser.add_argument('--image_dir', type=str, default='data/train',
                        help='Directory containing images')
    parser.add_argument('--train_anno', type=str, default='data/annos/annotations_train.json',
                        help='Path to training annotations file')
    parser.add_argument('--val_anno', type=str, default='data/annos/annotations_val.json',
                        help='Path to validation annotations file')
    parser.add_argument('--output_dir', type=str, default='data/kaggle_yolo',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--enhance', action='store_true', default=False,
                        help='Enable image enhancement (contrast stretching)')
    args = parser.parse_args()

    # Create output directory
    create_directory(args.output_dir)

    # Process training set
    print("\nProcessing training set...")
    convert_to_yolo_format(args.train_anno, args.image_dir,
                           args.output_dir, is_train=True, enhance=args.enhance)

    # Process validation set
    print("\nProcessing validation set...")
    convert_to_yolo_format(args.val_anno, args.image_dir,
                           args.output_dir, is_train=False, enhance=args.enhance)

    # Create dataset.yaml file
    yaml_content = f"""path: {os.path.abspath(args.output_dir)}
train: images/train
val: images/val

nc: {len(load_coco_annotations(args.train_anno)['categories'])}
names: {[cat['name'] for cat in load_coco_annotations(args.train_anno)['categories']]}
"""

    with open(os.path.join(args.output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(
        f"\nDataset conversion complete. YOLO format dataset saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
