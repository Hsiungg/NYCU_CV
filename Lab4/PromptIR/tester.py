import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, TestSpecificDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F
import torch.optim as optim
from utils.schedulers import LinearWarmupCosineAnnealingLR


class TTAWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, x):
        # Original
        pred = self.model(x)

        # Flip up and down
        x_ud = torch.flip(x, [-2])
        pred_ud = self.model(x_ud)
        pred_ud = torch.flip(pred_ud, [-2])

        # Rotate 90 degree counterclockwise
        x_r90 = torch.rot90(x, k=1, dims=[-2, -1])
        pred_r90 = self.model(x_r90)
        pred_r90 = torch.rot90(pred_r90, k=-1, dims=[-2, -1])

        # Rotate 90 degree and flip up and down
        x_r90ud = torch.rot90(x, k=1, dims=[-2, -1])
        x_r90ud = torch.flip(x_r90ud, [-2])
        pred_r90ud = self.model(x_r90ud)
        pred_r90ud = torch.flip(pred_r90ud, [-2])
        pred_r90ud = torch.rot90(pred_r90ud, k=-1, dims=[-2, -1])

        # Rotate 180 degree
        x_r180 = torch.rot90(x, k=2, dims=[-2, -1])
        pred_r180 = self.model(x_r180)
        pred_r180 = torch.rot90(pred_r180, k=-2, dims=[-2, -1])

        # Rotate 180 degree and flip
        x_r180ud = torch.rot90(x, k=2, dims=[-2, -1])
        x_r180ud = torch.flip(x_r180ud, [-2])
        pred_r180ud = self.model(x_r180ud)
        pred_r180ud = torch.flip(pred_r180ud, [-2])
        pred_r180ud = torch.rot90(pred_r180ud, k=-2, dims=[-2, -1])

        # Rotate 270 degree
        x_r270 = torch.rot90(x, k=3, dims=[-2, -1])
        pred_r270 = self.model(x_r270)
        pred_r270 = torch.rot90(pred_r270, k=-3, dims=[-2, -1])

        # Rotate 270 degree and flip
        x_r270ud = torch.rot90(x, k=3, dims=[-2, -1])
        x_r270ud = torch.flip(x_r270ud, [-2])
        pred_r270ud = self.model(x_r270ud)
        pred_r270ud = torch.flip(pred_r270ud, [-2])
        pred_r270ud = torch.rot90(pred_r270ud, k=-3, dims=[-2, -1])

        # Average all predictions
        return (pred + pred_ud + pred_r90 + pred_r90ud + pred_r180 + pred_r180ud + pred_r270 + pred_r270ud) / 8.0


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(
            num_blocks=[8, 12, 12, 16], num_refinement_blocks=8, decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        return super().load_from_checkpoint(checkpoint_path, **kwargs)


def test_Specific(net, dataset, use_tta=False):
    output_path = testopt.output_path + 'out_imgs/'
    npz_path = testopt.output_path + 'out_npz/'
    subprocess.check_output(['mkdir', '-p', output_path])
    subprocess.check_output(['mkdir', '-p', npz_path])

    # Dictionary to store results
    results_dict = {}

    testloader = DataLoader(dataset, batch_size=1,
                            pin_memory=True, shuffle=False, num_workers=0)

    # Create TTA model if needed
    model = TTAWrapper(net) if use_tta else net
    if not use_tta:
        model.eval()

    with torch.no_grad():
        for ([name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            restored = model(degrad_patch)
            # Convert tensor to numpy array
            restored_np = restored.cpu().numpy()[0]  # Remove batch dimension
            # Scale to 0-255 and convert to uint8
            restored_np = (restored_np * 255).astype(np.uint8)

            # Store in dictionary with filename as key
            results_dict[name[0]+'.png'] = restored_np

            # Also save as PNG for visualization
            save_image_tensor(restored, output_path + name[0] + '.png')

    # Save results as .npz file
    npz_file_path = os.path.join(npz_path, 'pred.npz')
    np.savez(npz_file_path, **results_dict)
    print("Testing completed. Results saved to: {}".format(output_path))
    print("NPZ file saved to: {}".format(npz_file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=4,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one, 4 for specific test')
    parser.add_argument('--output_path', type=str,
                        default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str,
                        default="/mnt/sda1/cv/checkpoints/deeper_model/", help='checkpoint save path')
    parser.add_argument('--ckpt_name', type=str,
                        default="promptir-epoch=63-val_psnr=30.07.ckpt", help='checkpoint save path')
    parser.add_argument('--test_path', type=str,
                        default="data/test/degraded", help='path to test images')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test-Time Augmentation')
    testopt = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_path + testopt.ckpt_name

    print("CKPT name : {}".format(ckpt_path))

    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.mode == 0:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(opt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path, name)
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif testopt.mode == 3:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")
    elif testopt.mode == 4:
        print('Start testing specific images...')
        specific_dataset = TestSpecificDataset(testopt)
        test_Specific(net, specific_dataset, use_tta=testopt.use_tta)
