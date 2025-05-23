import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from skimage import io

def check_image_shapes(image_dir):
    """Check shapes of all images in the directory"""
    print(f"\nChecking images in: {image_dir}")
    
    # Get all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend(list(Path(image_dir).rglob(f'*{ext}')))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Dictionary to store shape statistics
    shapes = {}
    channel_stats = {}
    format_stats = {}
    
    # Check each image
    for img_path in tqdm(image_files, desc="Checking images"):
        try:
            # Get file extension
            ext = img_path.suffix.lower()
            if ext not in format_stats:
                format_stats[ext] = 0
            format_stats[ext] += 1
            
            # Read image using skimage
            img = io.imread(str(img_path))
            
            # Get shape and channels
            shape = img.shape
            if len(shape) == 2:  # Grayscale
                channels = 1
            elif len(shape) == 3:  # Color or multi-channel
                channels = shape[2]
            else:
                channels = 1
            
            print(f"\nDetailed info for {img_path}:")
            print(f"  Format: {ext}")
            print(f"  Shape: {shape}")
            print(f"  Channels: {channels}")
            print(f"  Data type: {img.dtype}")
            print(f"  Value range: [{img.min()}, {img.max()}]")
            
            # Update channel statistics
            channel_key = f"{channels} channels"
            if channel_key not in channel_stats:
                channel_stats[channel_key] = {
                    'count': 0,
                    'examples': []
                }
            channel_stats[channel_key]['count'] += 1
            if len(channel_stats[channel_key]['examples']) < 3:
                channel_stats[channel_key]['examples'].append(str(img_path))
            
            # Update shape statistics
            shape_key = str(shape)
            if shape_key not in shapes:
                shapes[shape_key] = {
                    'count': 0,
                    'examples': [],
                    'is_grayscale': channels == 1
                }
            
            shapes[shape_key]['count'] += 1
            if len(shapes[shape_key]['examples']) < 3:
                shapes[shape_key]['examples'].append(str(img_path))
                
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
    
    # Print results
    print("\nFile Format Statistics:")
    print("-" * 50)
    for fmt, count in format_stats.items():
        percentage = (count / len(image_files)) * 100
        print(f"{fmt}: {count} files ({percentage:.1f}%)")
    
    print("\nChannel Statistics:")
    print("-" * 50)
    for channel, info in channel_stats.items():
        print(f"\n{channel}")
        print(f"Count: {info['count']}")
        print("Example files:")
        for example in info['examples']:
            print(f"  - {example}")
    
    print("\nImage Shape Statistics:")
    print("-" * 50)
    for shape, info in shapes.items():
        print(f"\nShape: {shape}")
        print(f"Count: {info['count']}")
        print(f"Type: {'Grayscale' if info['is_grayscale'] else 'Color'}")
        print("Example files:")
        for example in info['examples']:
            print(f"  - {example}")
    print("-" * 50)
    
    # Print summary
    print("\nSummary:")
    print(f"Total images: {len(image_files)}")
    print("Channel distribution:")
    for channel, info in channel_stats.items():
        percentage = (info['count'] / len(image_files)) * 100
        print(f"  {channel}: {info['count']} images ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Check image shapes in dataset')
    parser.add_argument('--image_dir', type=str, required=True,
                      help='Directory containing images to check')
    args = parser.parse_args()
    
    check_image_shapes(args.image_dir)

if __name__ == "__main__":
    main() 