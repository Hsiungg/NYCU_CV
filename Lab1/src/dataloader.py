from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import shutil
import torch


class ImageDataset(Dataset):
    def __init__(self, root_dir, mode='train', image_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

        if mode == 'train':
            self.dataset = datasets.ImageFolder(os.path.join(
                root_dir, 'train'), transform=self.transform)
        elif mode == 'val':
            self.dataset = datasets.ImageFolder(os.path.join(
                root_dir, 'val'), transform=self.transform)
        elif mode == 'test':
            source_dir = os.path.join(root_dir, 'test')
            target_dir = os.path.join(source_dir, "dummy")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for filename in os.listdir(source_dir):
                if filename.endswith('.jpg'):
                    source_path = os.path.join(source_dir, filename)
                    target_path = os.path.join(target_dir, filename)
                    shutil.move(source_path, target_path)
            self.dataset = datasets.ImageFolder(
                os.path.join(root_dir, 'test'), transform=self.transform, target_transform=lambda x: 0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_name = os.path.basename(self.dataset.imgs[idx][0])
        img_name_without_extension = os.path.splitext(img_name)[0]
        return {"pixel_values": img, "labels": torch.tensor(label), "image_name": img_name_without_extension}


def load_dataset(root_dir, mode='train'):
    return ImageDataset(root_dir, mode)
