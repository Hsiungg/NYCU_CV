import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml


def load_annotations(anno_path, dataset_type):
    """Load annotations in different formats"""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    if dataset_type == 'livecell':
        annotations = data['annotations']
        if isinstance(annotations, str):
            annotations = json.loads(annotations)

        # Convert annotations to proper format
        processed_annotations = []
        for anno in annotations:
            if isinstance(anno, (int, str)):
                processed_annotations.append({
                    'category_id': 1,  # Default category for LiveCell
                    'image_id': int(anno) if isinstance(anno, str) else anno
                })
            else:
                processed_annotations.append(anno)

        return {
            'annotations': processed_annotations,
            'categories': data['categories'],
            'images': data['images']
        }
    else:
        # Check if it's COCO format
        if isinstance(data, dict) and 'annotations' in data and 'categories' in data:
            return data

        # If it's a list of annotations
        if isinstance(data, list):
            # Create a COCO-like structure
            coco_format = {
                'annotations': data,
                'categories': [{'id': i, 'name': str(i)} for i in set(anno.get('category_id', 0) for anno in data)],
                'images': [{'id': i} for i in set(anno.get('image_id', 0) for anno in data)]
            }
            return coco_format

    raise ValueError(f"Unsupported annotation format in {anno_path}")


def analyze_dataset(image_dir, train_anno_path, val_anno_path, mask_path=None, dataset_type='kaggle'):
    """Analyze dataset metadata"""
    print("Loading annotations...")
    train_annotations = load_annotations(train_anno_path, dataset_type)
    val_annotations = load_annotations(val_anno_path, dataset_type)

    # Set image extension based on dataset type
    img_ext = '.tif' if dataset_type == 'livecell' else '.png'
    print(f"\nUsing image extension: {img_ext}")

    # Initialize statistics
    stats = {
        'train': {
            'total_images': len(train_annotations['images']),
            'total_annotations': len(train_annotations['annotations']),
            'image_sizes': [],
            'image_shapes': [],
            'categories': defaultdict(int),
            'missing_images': 0,
            'missing_masks': 0
        },
        'val': {
            'total_images': len(val_annotations['images']),
            'total_annotations': len(val_annotations['annotations']),
            'image_sizes': [],
            'image_shapes': [],
            'categories': defaultdict(int),
            'missing_images': 0,
            'missing_masks': 0
        }
    }

    # Create category mapping
    category_map = {cat['id']: cat['name']
                    for cat in train_annotations['categories']}

    # Load mask dictionary if provided
    mask_dict = None
    if mask_path and os.path.exists(mask_path):
        mask_dict = np.load(mask_path, allow_pickle=True).item()

    # Analyze training set
    print("\nAnalyzing training set...")
    for img in tqdm(train_annotations['images']):
        img_id = img['id']
        img_path = os.path.join(image_dir, f"{img_id}{img_ext}")

        # Check if image exists
        if not os.path.exists(img_path):
            stats['train']['missing_images'] += 1
            continue

        # Check if mask exists (only if mask_dict is provided)
        if mask_dict is not None and str(img_id) not in mask_dict:
            stats['train']['missing_masks'] += 1
            continue

        # Get image size and shape
        try:
            with Image.open(img_path) as img_file:
                stats['train']['image_sizes'].append(img_file.size)
                img_array = np.array(img_file)
                stats['train']['image_shapes'].append(img_array.shape)
        except Exception as e:
            print(f"Error reading image {img_id}: {e}")

    # Analyze validation set
    print("\nAnalyzing validation set...")
    for img in tqdm(val_annotations['images']):
        img_id = img['id']
        img_path = os.path.join(image_dir, f"{img_id}{img_ext}")

        # Check if image exists
        if not os.path.exists(img_path):
            stats['val']['missing_images'] += 1
            continue

        # Check if mask exists (only if mask_dict is provided)
        if mask_dict is not None and str(img_id) not in mask_dict:
            stats['val']['missing_masks'] += 1
            continue

        # Get image size and shape
        try:
            with Image.open(img_path) as img_file:
                stats['val']['image_sizes'].append(img_file.size)
                img_array = np.array(img_file)
                stats['val']['image_shapes'].append(img_array.shape)
        except Exception as e:
            print(f"Error reading image {img_id}: {e}")

    # Count annotations per category
    for anno in train_annotations['annotations']:
        if isinstance(anno, dict):
            cat_id = anno.get('category_id', 0)
        else:
            # If annotation is a string or number, use default category
            cat_id = 1  # Default category for LiveCell
        stats['train']['categories'][category_map.get(
            cat_id, str(cat_id))] += 1

    for anno in val_annotations['annotations']:
        if isinstance(anno, dict):
            cat_id = anno.get('category_id', 0)
        else:
            # If annotation is a string or number, use default category
            cat_id = 1  # Default category for LiveCell
        stats['val']['categories'][category_map.get(cat_id, str(cat_id))] += 1

    return stats


def analyze_shapes_and_ratios(stats, dataset_type):
    """Analyze image shapes and train/val ratios"""
    # Create output directory with dataset name
    output_dir = f'dataset_stats_{dataset_type}'
    os.makedirs(output_dir, exist_ok=True)

    # Analyze shapes
    train_shapes = np.array(stats['train']['image_shapes'])
    val_shapes = np.array(stats['val']['image_shapes'])

    if len(train_shapes) > 0 and len(val_shapes) > 0:
        # Get unique shapes and their counts
        train_unique_shapes, train_shape_counts = np.unique(
            train_shapes, axis=0, return_counts=True)
        val_unique_shapes, val_shape_counts = np.unique(
            val_shapes, axis=0, return_counts=True)

    # Calculate ratios
    total_images = stats['train']['total_images'] + \
        stats['val']['total_images']
    train_ratio = stats['train']['total_images'] / total_images
    val_ratio = stats['val']['total_images'] / total_images

    # Print shape information
    print("\nImage Shape Analysis:")
    print("\nTraining Set Shapes:")
    for shape, count in zip(train_unique_shapes, train_shape_counts):
        print(f"Shape {shape}: {count} images")

    print("\nValidation Set Shapes:")
    for shape, count in zip(val_unique_shapes, val_shape_counts):
        print(f"Shape {shape}: {count} images")

    print("\nTrain/Val Split Ratio:")
    print(f"Training set: {train_ratio:.2%}")
    print(f"Validation set: {val_ratio:.2%}")

    # Plot shape distribution
    plt.figure(figsize=(15, 6))

    # Plot training set shape distribution
    plt.subplot(1, 2, 1)
    plt.bar(range(len(train_shape_counts)), train_shape_counts, color='blue')
    plt.title('Training Set Shape Distribution')
    plt.xlabel('Shape Index')
    plt.ylabel('Count')
    plt.xticks(range(len(train_shape_counts)),
               [f"{shape}" for shape in train_unique_shapes],
               rotation=45, ha='right')

    # Plot validation set shape distribution
    plt.subplot(1, 2, 2)
    plt.bar(range(len(val_shape_counts)), val_shape_counts, color='red')
    plt.title('Validation Set Shape Distribution')
    plt.xlabel('Shape Index')
    plt.ylabel('Count')
    plt.xticks(range(len(val_shape_counts)),
               [f"{shape}" for shape in val_unique_shapes],
               rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/shape_distribution.png')
    plt.close()

    # Plot train/val ratio
    plt.figure(figsize=(8, 6))
    plt.pie([train_ratio, val_ratio],
            labels=['Training Set', 'Validation Set'],
            colors=['blue', 'red'],
            autopct='%1.1f%%')
    plt.title('Train/Val Split Ratio')
    plt.savefig(f'{output_dir}/train_val_ratio.png')
    plt.close()


def plot_statistics(stats, dataset_type):
    """Plot dataset statistics"""
    # Create output directory with dataset name
    output_dir = f'dataset_stats_{dataset_type}'
    os.makedirs(output_dir, exist_ok=True)

    # Plot image size distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    train_sizes = np.array(stats['train']['image_sizes'])
    if len(train_sizes) > 0:
        train_sizes = np.array([(w, h) for w, h in train_sizes])
        plt.scatter(train_sizes[:, 0], train_sizes[:, 1],
                    alpha=0.5, label='Train', color='blue')
    plt.title('Training Set Image Sizes')
    plt.xlabel('Width')
    plt.ylabel('Height')
    if len(train_sizes) > 0:
        plt.legend()

    plt.subplot(1, 2, 2)
    val_sizes = np.array(stats['val']['image_sizes'])
    if len(val_sizes) > 0:
        val_sizes = np.array([(w, h) for w, h in val_sizes])
        plt.scatter(val_sizes[:, 0], val_sizes[:, 1],
                    alpha=0.5, label='Val', color='red')
    plt.title('Validation Set Image Sizes')
    plt.xlabel('Width')
    plt.ylabel('Height')
    if len(val_sizes) > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/image_sizes.png')
    plt.close()

    # Plot category distribution
    plt.figure(figsize=(15, 6))
    categories = sorted(set(list(stats['train']['categories'].keys()) +
                            list(stats['val']['categories'].keys())))
    if categories:  # Only plot if there are categories
        x = np.arange(len(categories))
    width = 0.35

    train_counts = [stats['train']['categories'].get(
        cat, 0) for cat in categories]
    val_counts = [stats['val']['categories'].get(cat, 0) for cat in categories]

    plt.bar(x - width/2, train_counts, width, label='Train', color='blue')
    plt.bar(x + width/2, val_counts, width, label='Val', color='red')
    plt.xlabel('Categories')
    plt.ylabel('Number of Annotations')
    plt.title('Category Distribution')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_distribution.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset metadata')
    parser.add_argument('--dataset', choices=['livecell', 'kaggle'], required=True,
                        help='choose which dataset to use')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='path to image directory')
    parser.add_argument('--train_anno', type=str, required=True,
                        help='path to training annotations')
    parser.add_argument('--val_anno', type=str, required=True,
                        help='path to validation annotations')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='path to mask file (optional)')
    args = parser.parse_args()

    # Analyze dataset
    print("Starting dataset analysis...")
    stats = analyze_dataset(args.image_dir, args.train_anno,
                            args.val_anno, args.mask_path, args.dataset)

    # Print statistics
    print("\nDataset Statistics:")
    print("\nTraining Set:")
    print(f"Total Images: {stats['train']['total_images']}")
    print(f"Total Annotations: {stats['train']['total_annotations']}")
    print(f"Missing Images: {stats['train']['missing_images']}")
    if args.mask_path:
        print(f"Missing Masks: {stats['train']['missing_masks']}")
    print("\nCategories in Training Set:")
    total_train = sum(stats['train']['categories'].values())
    for cat, count in stats['train']['categories'].items():
        print(f"{cat}: {count} ({count/total_train:.2%})")

    print("\nValidation Set:")
    print(f"Total Images: {stats['val']['total_images']}")
    print(f"Total Annotations: {stats['val']['total_annotations']}")
    print(f"Missing Images: {stats['val']['missing_images']}")
    if args.mask_path:
        print(f"Missing Masks: {stats['val']['missing_masks']}")
    print("\nCategories in Validation Set:")
    total_val = sum(stats['val']['categories'].values())
    for cat, count in stats['val']['categories'].items():
        print(f"{cat}: {count} ({count/total_val:.2%})")

    # Analyze shapes and ratios
    print("\nAnalyzing image shapes and ratios...")
    analyze_shapes_and_ratios(stats, args.dataset)

    # Plot statistics
    print("\nGenerating plots...")
    plot_statistics(stats, args.dataset)
    print(f"Plots saved in 'dataset_stats_{args.dataset}' directory")


if __name__ == "__main__":
    main()
