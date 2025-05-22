import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, random_split
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


def cutmix(clean_patch, degrad_patch, clean_patch2, degrad_patch2, alpha=1.0):
    """
    Perform CutMix data augmentation
    Args:
        clean_patch: First clean image
        degrad_patch: First degraded image
        clean_patch2: Second clean image
        degrad_patch2: Second degraded image
        alpha: Beta distribution parameter
    Returns:
        mixed_clean: Mixed clean image
        mixed_degrad: Mixed degraded image
        lam: Mixing ratio
    """
    # Generate random mixing ratio
    lam = np.random.beta(alpha, alpha)

    # Get image dimensions
    H, W = clean_patch.shape[1:]

    # Generate random crop region
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Randomly select center point of crop region
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Ensure crop region is within image bounds
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Perform mixing
    mixed_clean = clean_patch.clone()
    mixed_degrad = degrad_patch.clone()

    mixed_clean[:, bby1:bby2,
                bbx1:bbx2] = clean_patch2[:, bby1:bby2, bbx1:bbx2]
    mixed_degrad[:, bby1:bby2,
                 bbx1:bbx2] = degrad_patch2[:, bby1:bby2, bbx1:bbx2]

    # Adjust mixing ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return mixed_clean, mixed_degrad, lam


def split_dataset(dataset, val_ratio=0.2, seed=42):
    """
    Split dataset into training and validation sets
    Args:
        dataset: Dataset to split
        val_ratio: Validation set ratio
        seed: Random seed
    Returns:
        train_dataset, val_dataset
    """
    # Group samples by degradation type
    derain_pairs = []
    desnow_pairs = []

    for pair in dataset.sample_pairs:
        if pair[3] == 3:  # derain
            derain_pairs.append(pair)
        elif pair[3] == 6:  # desnow
            desnow_pairs.append(pair)

    # Calculate validation size for each type
    derain_val_size = int(len(derain_pairs) * val_ratio)
    desnow_val_size = int(len(desnow_pairs) * val_ratio)

    # Set random seed for reproducibility
    random.seed(seed)

    # Randomly select validation samples for each type
    derain_val_indices = random.sample(
        range(len(derain_pairs)), derain_val_size)
    desnow_val_indices = random.sample(
        range(len(desnow_pairs)), desnow_val_size)

    # Create validation set
    val_pairs = []
    for idx in derain_val_indices:
        val_pairs.append(derain_pairs[idx])
    for idx in desnow_val_indices:
        val_pairs.append(desnow_pairs[idx])

    # Create training set
    train_pairs = []
    for i, pair in enumerate(derain_pairs):
        if i not in derain_val_indices:
            train_pairs.append(pair)
    for i, pair in enumerate(desnow_pairs):
        if i not in desnow_val_indices:
            train_pairs.append(pair)

    # Create new datasets
    train_dataset = PromptTrainDataset(
        dataset.root_dir,
        patch_size=dataset.patch_size,
        is_train=True,
        use_cutmix=dataset.use_cutmix,
        cutmix_prob=dataset.cutmix_prob
    )
    train_dataset.sample_pairs = train_pairs

    val_dataset = PromptTrainDataset(
        dataset.root_dir,
        patch_size=dataset.patch_size,
        is_train=False,  # Validation set should not use augmentation
        use_cutmix=False,  # Disable CutMix for validation
        cutmix_prob=0.0
    )
    val_dataset.sample_pairs = val_pairs

    print(f"Training set: {len(train_pairs)} samples")
    print(f"Validation set: {len(val_pairs)} samples")
    print(f"Derain validation samples: {len(derain_val_indices)}")
    print(f"Desnow validation samples: {len(desnow_val_indices)}")

    return train_dataset, val_dataset


class PromptTrainDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, is_train=True, use_cutmix=True, cutmix_prob=0.5):
        super().__init__()
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, "clean")
        self.degraded_dir = os.path.join(root_dir, "degraded")
        self.patch_size = patch_size
        self.is_train = is_train
        self.use_cutmix = use_cutmix and is_train  # Only enable CutMix for training
        self.cutmix_prob = cutmix_prob

        self.toTensor = ToTensor()
        if is_train:
            self.crop_transform = Compose([
                ToPILImage(),
                RandomCrop(patch_size),
            ])
        else:
            self.crop_transform = None

        self.sample_pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        for fname in os.listdir(self.clean_dir):
            if not fname.endswith(".png"):
                continue
            base = fname.replace("_clean", "")
            clean_path = os.path.join(self.clean_dir, fname)
            degraded_path = os.path.join(self.degraded_dir, base)
            if os.path.exists(degraded_path):
                de_id = 3 if "rain" in fname else 6 if "snow" in fname else -1
                if de_id >= 0:
                    pairs.append((clean_path, degraded_path,
                                 fname.split(".")[0], de_id))
        print(f"Total samples loaded: {len(pairs)}")
        return pairs

    def _crop_patch(self, img_1, img_2):
        if self.is_train:
            H, W = img_1.shape[:2]
            ind_H = random.randint(0, H - self.patch_size)
            ind_W = random.randint(0, W - self.patch_size)
            patch_1 = img_1[ind_H:ind_H + self.patch_size,
                            ind_W:ind_W + self.patch_size]
            patch_2 = img_2[ind_H:ind_H + self.patch_size,
                            ind_W:ind_W + self.patch_size]
            return patch_1, patch_2
        else:
            return img_1, img_2

    def _add_gaussian_noise(self, img, sigma_range=(0, 15)):
        """
        Add random Gaussian noise to the image
        Args:
            img: Input image (numpy array)
            sigma_range: Range of noise standard deviation (min, max)
        Returns:
            Noisy image
        """
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, img.shape)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy_img

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        clean_path, degraded_path, clean_name, de_id = self.sample_pairs[idx]

        clean_img = np.array(Image.open(clean_path).convert("RGB"))
        degrad_img = np.array(Image.open(degraded_path).convert("RGB"))

        if self.is_train:
            degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

            # Convert to tensors for CutMix
            # This converts to CHW format and normalizes to [0,1]
            clean_patch = self.toTensor(clean_patch)
            # This converts to CHW format and normalizes to [0,1]
            degrad_patch = self.toTensor(degrad_patch)

            # Apply CutMix
            if self.use_cutmix and random.random() < self.cutmix_prob:
                # Find another image with the same degradation type
                same_degradation_pairs = [(i, pair) for i, pair in enumerate(self.sample_pairs)
                                          if pair[3] == de_id and i != idx]

                if same_degradation_pairs:  # Only proceed if we found a matching image
                    idx2, (clean_path2, degraded_path2, _,
                           _) = random.choice(same_degradation_pairs)

                    clean_img2 = np.array(
                        Image.open(clean_path2).convert("RGB"))
                    degrad_img2 = np.array(Image.open(
                        degraded_path2).convert("RGB"))

                    degrad_patch2, clean_patch2 = self._crop_patch(
                        degrad_img2, clean_img2)

                    # Convert to tensors
                    clean_patch2 = self.toTensor(clean_patch2)
                    degrad_patch2 = self.toTensor(degrad_patch2)

                    # Perform CutMix
                    clean_patch, degrad_patch, _ = cutmix(
                        clean_patch, degrad_patch, clean_patch2, degrad_patch2)

            # Convert to numpy for augmentation and noise
            if clean_patch.is_cuda:
                clean_patch = clean_patch.cpu()
            if degrad_patch.is_cuda:
                degrad_patch = degrad_patch.cpu()

            # Convert back to [0, 255] range for numpy operations
            clean_patch = (clean_patch * 255).permute(1,
                                                      2, 0).numpy().astype(np.uint8)
            degrad_patch = (degrad_patch * 255).permute(1,
                                                        2, 0).numpy().astype(np.uint8)

            # Randomly add Gaussian noise
            if random.random() < 0.5:  # 50% probability to add noise
                degrad_patch = self._add_gaussian_noise(degrad_patch)

            # Apply augmentation after CutMix and noise
            degrad_patch, clean_patch = random_augmentation(
                degrad_patch, clean_patch)

            # Convert back to tensor and ensure CHW format
            clean_patch = torch.from_numpy(
                clean_patch).permute(2, 0, 1).float() / 255.0
            degrad_patch = torch.from_numpy(
                degrad_patch).permute(2, 0, 1).float() / 255.0
        else:
            degrad_patch = self.toTensor(degrad_img)
            clean_patch = self.toTensor(clean_img)

        return [clean_name, de_id], degrad_patch, clean_patch


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise *
                              self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(
            np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(
            clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_, tile=128, tile_overlap=0):
        sigma_dict = {0: 0, 1: 15, 2: 25, 3: 50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)
                  ].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain", addnoise=False, sigma=None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise *
                              self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path +
                         'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path +
                         'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(
            np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(
            np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(
            clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception(
                    'The input directory does not contain any image files')
            self.degraded_ids = [os.path.join(root, id_) for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(
            np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = os.path.basename(self.degraded_ids[idx]).split('.')[0]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
