import os
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim, ms_ssim
import torchvision.models as models
import torch.nn.functional as F

from utils.dataset_utils import PromptTrainDataset, split_dataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        try:
            # Try new version first
            return 1 - ssim(pred, target, data_range=1.0, reduction='mean')
        except TypeError:
            # Fall back to old version
            return 1 - ssim(pred, target, data_range=1.0, size_average=True)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class MS_SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Ensure inputs are in valid range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Add small epsilon to prevent division by zero
        pred = pred + 1e-8
        target = target + 1e-8

        try:
            # Try new version first
            return 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)
        except Exception as e:
            print(f"MS-SSIM calculation failed: {str(e)}")
            # Fall back to regular SSIM if MS-SSIM fails
            return 1 - ssim(pred, target, data_range=1.0, size_average=True)


class CombinedLoss(nn.Module):
    def __init__(self, w_charbonnier=0.85, w_ssim=0.15, mode='train'):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.w_charbonnier = w_charbonnier
        self.w_ssim = w_ssim
        self.mode = mode

    def forward(self, pred, target):
        loss_charbonnier = self.charbonnier(pred, target)
        loss_ssim = self.ssim(pred, target)
        return self.w_charbonnier * loss_charbonnier + self.w_ssim * loss_ssim


class PromptIRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = PromptIR(decoder=True)
        self.train_loss_fn = CombinedLoss(
            w_charbonnier=0.85, w_ssim=0.15, mode='train')
        self.val_loss_fn = CombinedLoss(
            w_charbonnier=0.85, w_ssim=0.15, mode='val')
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.train_loss_fn(restored, clean_patch)

        # Log training metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate",
                 self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("step", int(self.global_step), prog_bar=True)
        self.log("epoch", int(self.current_epoch), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.val_loss_fn(restored, clean_patch)

        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)

        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)
        self.log("val_ssim", ssim, prog_bar=True)
        self.log("learning_rate",
                 self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("step", int(self.global_step), prog_bar=True)
        self.log("epoch", int(self.current_epoch), prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=0.1)

        if self.args.scheduler == 'cosine_warmup':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=20,
                max_epochs=self.args.epochs,
                eta_min=1e-6,
            )
            return [optimizer], [scheduler]

        elif self.args.scheduler == 'cosine_restart':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=10,  # First restart epoch
                T_mult=2,  # Multiply T_0 by this factor after each restart
                eta_min=2e-6  # Minimum learning rate
            )
            return [optimizer], [scheduler]

        elif self.args.scheduler == 'one_cycle':
            # Use a fixed number of steps per epoch
            steps_per_epoch = int((3200 * 0.8) // self.args.batch_size)
            total_steps = self.args.epochs * steps_per_epoch

            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.args.lr,
                total_steps=total_steps,
                pct_start=0.3,  # 30% of training for warmup
                div_factor=25,  # Initial lr = max_lr/25
                final_div_factor=1e3  # Final lr = initial_lr/1e4
            )
            return [optimizer], [scheduler]

        elif self.args.scheduler == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='max',  # Monitor validation PSNR
                factor=0.5,  # Reduce lr by half
                patience=5  # Number of epochs to wait before reducing lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_psnr"  # Monitor validation PSNR
                }
            }

        elif self.args.scheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[30, 60, 90],  # Reduce lr at these epochs
                gamma=0.5  # Multiply lr by this factor
            )
            return [optimizer], [scheduler]

        else:
            return [optimizer]  # No scheduler


def main():
    print("Options")
    print(opt)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # Create dataset with CutMix enabled
    dataset = PromptTrainDataset(
        opt.training_root_dir,
        patch_size=opt.patch_size,
        use_cutmix=True,
        cutmix_prob=0.5  # 50% probability of using CutMix
    )
    # Use a different random seed for dataset splitting
    train_dataset, val_dataset = split_dataset(
        dataset, val_ratio=0.2, seed=123)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    # Try to load checkpoint if specified, otherwise start with new model
    if opt.resume:
        try:
            print(f"Attempting to load checkpoint from {opt.resume}")
            model = PromptIRModel.load_from_checkpoint(
                checkpoint_path=opt.resume,
                args=opt
            )
            print("Successfully loaded checkpoint")
        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            print("Starting training with new model")
            model = PromptIRModel(opt)
    else:
        model = PromptIRModel(opt)

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename='promptir-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=5,
        monitor='val_psnr',
        mode='max',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        precision=16,
        accelerator="gpu",
        gradient_clip_val=1.0,
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
