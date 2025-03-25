"""Build training pypline with Trainer from transformers"""
import argparse
import os
import math

from test import test_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from torch.optim import AdamW

from utils import count_parameters, compute_metrics, same_seeds
from dataloader import load_dataset, MixUpCollator
from model import ResNetForClassification


def train_model(args):   # pylint: disable=redefined-outer-name
    """Set up training parameters and pipeline for model"""
    print(
        f"Training ResNet on {args.data_root} with batch_size={args.batch_size},\
        lr={args.lr}, epochs={args.epochs}"
    )

    # Load dataset
    train_dataset = load_dataset(args.data_root, mode='train')
    val_dataset = load_dataset(args.data_root, mode='val')
    num_classes = len(train_dataset.class_names)
    # Create model (Based on ResNet)
    print(f"Number of classes: {num_classes}")
    model = ResNetForClassification(num_classes)
    print(f"Total Parameters: {count_parameters(model)}M")

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = steps_per_epoch * 10

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=3,
    )
    print(f"Step per epoch : {steps_per_epoch}")

    # Freeze some layers if desired
    # for param in model.model.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.model.layer2.parameters():
    #     param.requires_grad = False
    logs_path = os.path.join("./logs", args.run_name)
    # Training args
    same_seeds(4444)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        dataloader_num_workers=4,
        logging_dir=logs_path,
        logging_steps=steps_per_epoch,
        load_best_model_at_end=True,
        report_to="tensorboard",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        max_grad_norm=1.0,
        seed=4444
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_threshold=0.01,
        early_stopping_patience=20
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        # add Mixup for augmentation
        data_collator=MixUpCollator(num_classes=num_classes, alpha=0.8),
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
    save_path = os.path.join(args.output_dir, 'best_model')
    trainer.save_model(save_path)
    # Test the model
    save_name = os.path.join(save_path, "model.safetensors")
    test_model(save_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet using Hugging Face Trainer")
    parser.add_argument("--data_root", type=str, default="data/",
                        help="Path to dataset root directory")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float,
                        default=0.03, help="Weight decay for optimizer")
    parser.add_argument("--output_dir", type=str,
                        default="./output_model/res50", help="Directory to save model")
    parser.add_argument("--num_classes", type=int,
                        default=100, help="Number of classes")
    parser.add_argument("--run_name", type=str,
                        default="res50", help="log run name")
    args = parser.parse_args()
    train_model(args)
