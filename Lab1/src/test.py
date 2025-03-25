"""testing trained model using testing dataset"""
import os
import argparse
import csv
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from model import ResNetForClassification
from dataloader import load_dataset


def test_model(model_path, args):  # pylint: disable=redefined-outer-name
    """build a pipeline for testing trained model"""
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetForClassification(num_classes=args.num_classes)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    # Load the test dataset
    test_dataset = load_dataset(args.data_root, mode='test')

    # Open CSV file to write predictions
    csv_path = os.path.join(args.output_dir, 'prediction.csv')
    with open(csv_path, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'pred_label'])

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # Iterate over test dataset
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["pixel_values"]
            image_names = batch['image_name']

            images = images.to(device)
            # Get predictions
            with torch.no_grad():
                outputs = model(images)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=1)

            # Write predictions to CSV file
            for image_name, pred in zip(image_names, preds):
                writer.writerow([image_name, pred.item()])

    print(f"Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet using Hugging Face Trainer")
    parser.add_argument("--data_root", type=str, default="data/",
                        help="Path to dataset root directory")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--output_dir", type=str,
                        default="./output_model/res34", help="Directory to save model")
    parser.add_argument("--num_classes", type=int,
                        default=100, help="Number of classes")
    parser.add_argument("--model_path", type=str,
                        default="output_model/res34/best_model/model.safetensors",
                        help="path to model")
    args = parser.parse_args()
    test_model(args.model_path, args)
