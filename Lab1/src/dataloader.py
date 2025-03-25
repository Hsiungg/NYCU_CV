"""Adjust dataset to meet our needs"""
import os
import shutil

from torchvision import transforms
from torchvision.transforms.v2 import MixUp
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    """Preprocess data using tansform and split into train, valid, and test"""

    def __init__(self, root_dir, mode='train', image_size=(224, 224)):
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.train_transform = transforms.Compose([
            # Randomly crop and resize to 224x224
            transforms.RandomResizedCrop((384, 384)),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandAugment(
                num_ops=2, magnitude=20),  # Apply RandAugment
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),  # Normalize
        ])
        self.test_transform = transforms.Compose([
            # Resize the image to 256x256 (without augmentation)
            transforms.Resize((412, 412)),
            transforms.CenterCrop((384, 384)),  # Crop the center to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # Normalize
        ])

        self.image_paths = []
        self.labels = []

        if self.mode != 'test':
            self.class_names = sorted(os.listdir(
                os.path.join(root_dir, mode)), key=int)
            for label, class_name in enumerate(self.class_names):
                class_folder = os.path.join(root_dir, mode, class_name)
                if os.path.isdir(class_folder):
                    for filename in os.listdir(class_folder):
                        # Ensure valid image formats
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            self.image_paths.append(
                                os.path.join(class_folder, filename))
                            self.labels.append(label)

        else:
            source_dir = os.path.join(root_dir, 'test')
            target_dir = os.path.join(source_dir, "dummy")

            # Create a 'dummy' folder if it doesn't exist
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Move all images in the test directory to the 'dummy' folder
            for filename in os.listdir(source_dir):
                # Ensure valid image formats
                if filename.endswith('.jpg'):
                    source_path = os.path.join(source_dir, filename)
                    target_path = os.path.join(target_dir, filename)
                    shutil.move(source_path, target_path)

            self.image_paths = [os.path.join(
                target_dir, filename) for filename in os.listdir(target_dir)]
            self.labels = [0] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img_file = Image.open(img_path).convert('RGB')

        if self.mode == 'train':
            img_file = self.train_transform(img_file)
        else:
            img_file = self.test_transform(img_file)

        img_name = os.path.basename(img_path)
        img_name_without_extension = os.path.splitext(img_name)[0]
        return {"pixel_values": img_file, "labels": torch.tensor(label),
                "image_name": img_name_without_extension}


class MixUpCollator:
    """Implement MixUp data augmentation by changing the collator"""

    def __init__(self, num_classes, alpha=0.5):
        self.mixup = MixUp(alpha=alpha, num_classes=num_classes)

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])

        pixel_values, labels = self.mixup(pixel_values, labels)

        return {"pixel_values": pixel_values, "labels": labels}


def load_dataset(root_dir, mode='train'):
    """Load a dataset from root directory"""
    return ImageDataset(root_dir, mode)


if __name__ == "__main__":
    dataset = load_dataset('data/', mode='train')
    sample = dataset[1]
    print(sample["pixel_values"].shape)
    print(sample["labels"])
    print(sample["image_name"])
    if isinstance(sample["pixel_values"], torch.Tensor):
        img = sample["pixel_values"]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).numpy()

        plt.imshow(img_np)
        plt.axis('off')
        plt.show()
