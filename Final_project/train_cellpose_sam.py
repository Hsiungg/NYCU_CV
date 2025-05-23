import os
import numpy as np
import json
from PIL import Image
from pathlib import Path
from cellpose import models, core, io, plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime
from custom_train import train_seg_with_callback
from torch.utils.tensorboard import SummaryWriter
import argparse

def load_coco_annotations(anno_path):
    """Load COCO format annotations"""
    with open(anno_path, 'r') as f:
        return json.load(f)

def get_image_ids_from_coco(coco_data):
    """Extract image IDs from COCO format"""
    return [img['id'] for img in coco_data['images']]

def prepare_data(image_dir, mask_path, train_anno_path, val_anno_path):
    """Prepare training and validation data using COCO format annotations"""
    # Load masks
    print(f"Loading masks from {mask_path}...")
    mask_dict = np.load(mask_path, allow_pickle=True).item()
    
    # Load COCO annotations
    print(f"Loading COCO annotations...")
    train_annotations = load_coco_annotations(train_anno_path)
    val_annotations = load_coco_annotations(val_anno_path)
    
    # Get list of actual image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png'))]
    
    # Create a mapping from filenames to image IDs
    filename_to_id = {}
    for img in train_annotations['images']:
        base_name = img['file_name'].split('.')[0]  # Remove extension
        for filename in image_files:
            if filename.startswith(base_name):
                filename_to_id[filename] = img['id']
                break
    
    # Add validation images to filename_to_id mapping
    for img in val_annotations['images']:
        base_name = img['file_name'].split('.')[0]  # Remove extension
        for filename in image_files:
            if filename.startswith(base_name):
                filename_to_id[filename] = img['id']
                break
    
    # Get image IDs from COCO annotations
    train_ids = [img['id'] for img in train_annotations['images'] if any(f.startswith(img['file_name'].split('.')[0]) for f in image_files)]
    val_ids = [img['id'] for img in val_annotations['images'] if any(f.startswith(img['file_name'].split('.')[0]) for f in image_files)]
    
    # Prepare data arrays
    train_images = []
    train_masks = []
    val_images = []
    val_masks = []
    
    # Process training data
    print("\nProcessing training data...")
    successful_loads = 0
    for image_id in tqdm(train_ids):
        # Find the corresponding filename
        filename = next((f for f in image_files if filename_to_id.get(f) == image_id), None)
        if filename is None:
            continue
            
        img_path = os.path.join(image_dir, filename)
        
        try:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        except Exception as e:
            continue
        
        # Use image ID to find mask
        if str(image_id) not in mask_dict:
            continue
            
        mask = mask_dict[str(image_id)]
        
        train_images.append(img_np)
        train_masks.append(mask)
        successful_loads += 1
        
    
    # Process validation data
    print("\nProcessing validation data...")
    successful_loads = 0
    for image_id in tqdm(val_ids):
        # Find the corresponding filename
        filename = next((f for f in image_files if filename_to_id.get(f) == image_id), None)
        if filename is None:
            continue
            
        img_path = os.path.join(image_dir, filename)
        
        try:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        except Exception as e:
            continue
        
        # Use image ID to find mask
        if str(image_id) not in mask_dict:
            continue
            
        mask = mask_dict[str(image_id)]
        
        val_images.append(img_np)
        val_masks.append(mask)
        successful_loads += 1
        
        if successful_loads % 50 == 0:
            print(f"Successfully loaded {successful_loads} validation images")
    
    print(f"\nFinal training set size: {len(train_images)}")
    print(f"Final validation set size: {len(val_images)}")
    
    if len(train_images) == 0 or len(train_masks) == 0:
        raise ValueError("No training data was loaded! Check the paths and file formats.")
    if len(val_images) == 0 or len(val_masks) == 0:
        raise ValueError("No validation data was loaded! Check the paths and file formats.")
    
    return train_images, train_masks, val_images, val_masks

def train_model(train_images, train_masks, val_images, val_masks, 
                model_name="cellpose_sam_model",
                n_epochs=20,
                learning_rate=1e-5,
                weight_decay=0.1,
                batch_size=2,
                save_path="checkpoints",
                save_every=5,
                use_tensorboard=False):
    """Train Cellpose-SAM model with wandb logging"""
    # Create a descriptive run name
    run_name = f"{model_name}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    
    # Initialize wandb
    wandb.init(
        project="cellpose-sam-training",
        name=run_name,
        config={
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_size": len(train_images),
            "val_size": len(val_images)
        }
    )
    
    io.logger_setup() # run this to get printing of progress
    
    #Check if colab notebook instance has GPU access
    if core.use_gpu()==False:
        raise ImportError("No GPU access, change your runtime")
    
    # Initialize model
    model = models.CellposeModel(gpu=True)
    
    # Train model
    print(f"\nStarting training for {n_epochs} epochs...")
    
    # Create a custom callback for wandb logging
    class WandbCallback:
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
            self.learning_rates = []
            
        def on_epoch_end(self, epoch, train_loss, val_loss, learning_rate):
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate
            })
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(learning_rate)

    # Create a custom callback for TensorBoard logging
    class TensorBoardCallback:
        def __init__(self, log_dir='runs'):
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
            self.train_losses = []
            self.val_losses = []
            self.learning_rates = []
            
        def on_epoch_end(self, epoch, train_loss, val_loss, learning_rate):
            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_rate', learning_rate, epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(learning_rate)
            
        def close(self):
            self.writer.close()
    
    # Choose callback based on preference
    if use_tensorboard:
        callback = TensorBoardCallback()
    else:
        callback = WandbCallback()
    
    new_model_path, train_losses, test_losses = train_seg_with_callback(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=val_images,
        test_labels=val_masks,
        batch_size=batch_size,
        n_epochs=n_epochs,
        min_train_masks=0,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_images)),
        save_every=save_every,
        save_path=save_path,
        model_name=model_name,
        callback=callback,
        validate_every=1
    )
    
    # Log final metrics
    if use_tensorboard:
        callback.close()  # Close TensorBoard writer
    else:
        wandb.log({
            "final_train_loss": train_losses[-1],
            "final_val_loss": test_losses[-1]
        })
        wandb.finish()
    
    return new_model_path, train_losses, test_losses

def main():
    # Set paths
    parser = argparse.ArgumentParser(description='Train Cellpose-SAM model')
    parser.add_argument('--cell_type', type=str, default=None,
                      help='Cell type to train on (e.g., a172, bt474, etc.)')
    args = parser.parse_args()

    if args.cell_type:
        # Use cell-type specific directory
        image_dir = f"data/livecell_processed/{args.cell_type}/images"
        mask_path = f"data/livecell_processed/{args.cell_type}/all_instance_masks.npy"
        train_anno_path = f"data/LIVECell_dataset_2021/annotations/LIVECell_single_cells/{args.cell_type}/livecell_{args.cell_type}_train.json"
        val_anno_path = f"data/LIVECell_dataset_2021/annotations/LIVECell_single_cells/{args.cell_type}/livecell_{args.cell_type}_val.json"
    else:
        # Use full dataset
    image_dir = "data/livecell_processed/images"
    mask_path = "data/livecell_processed/all_instance_masks.npy"
    train_anno_path = "data/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
    val_anno_path = "data/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json"
    
    # Prepare data
    print("Preparing data...")
    train_images, train_masks, val_images, val_masks = prepare_data(
        image_dir, mask_path, train_anno_path, val_anno_path
    )
    
    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")
    
    # Train model
    print("Training model...")
    model_path, train_losses, test_losses = train_model(
        train_images, train_masks, val_images, val_masks,
        model_name=f"cellpose_sam_model_{args.cell_type}" if args.cell_type else "cellpose_sam_model",
        n_epochs=5,
        learning_rate=1e-5,
        weight_decay=0.1,
        batch_size=1,
        use_tensorboard=False
    )
    
    print(f"Model saved to: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    main() 