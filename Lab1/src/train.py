from dataloader import load_dataset
from model import ResNetForClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import argparse
import os
import math
from utils import count_parameters, compute_metrics, same_seeds
from test import test_model
from torch.utils.tensorboard import SummaryWriter


def train_model(args):
    print(
        f"Training ResNet on {args.data_root} with batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}"
    )

    # Load dataset
    train_dataset = load_dataset(args.data_root, mode='train')
    val_dataset = load_dataset(args.data_root, mode='val')
    num_classes = len(train_dataset.dataset.classes)

    # Create model (Based on ResNet)
    print(f"Number of classes: {num_classes}")
    model = ResNetForClassification(num_classes)
    print(f"Total Parameters: {count_parameters(model)}M")

    # Define optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, div_factor=100, final_div_factor=10,
                           epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    # Freeze some layers if desired
    # for param in model.model.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.model.layer2.parameters():
    #     param.requires_grad = False

    # Training args
    same_seeds(42)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=4,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=10,
        load_best_model_at_end=True,
        report_to="tensorboard",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        seed=42
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_threshold=0.01,
        early_stopping_patience=5
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, scheduler),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    train_dataloader = trainer.get_train_dataloader()
    train_dataloader.pin_memory = True
    print(trainer.get_train_dataloader())
    # Train the model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    trainer.train()
    # Save the best model
    save_name = os.path.join(args.output_dir, "best_model.pth")
    torch.save(model.state_dict(), save_name)
    # Test the model
    test_model(save_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet using Hugging Face Trainer")
    parser.add_argument("--data_root", type=str, default="data/",
                        help="Path to dataset root directory")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--output_dir", type=str,
                        default="./output_model/res50", help="Directory to save model")
    parser.add_argument("--num_classes", type=int,
                        default=100, help="Number of classes")
    args = parser.parse_args()
    train_model(args)
