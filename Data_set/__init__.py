"""
preload the dataset and dataloader
"""

# tiny_imagenet_loader.py
import os
import torch
import pandas as pd
import polars as pl
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

current_path = os.getcwd()
data_dir = os.path.join(current_path,r'Data_set',r'tiny-imagenet-200')

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'train')
            self.images = []
            self.labels = []

            for class_id in os.listdir(self.data_dir):
                class_dir = os.path.join(self.data_dir, class_id, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            self.images.append(os.path.join(class_dir, img_name))
                            self.labels.append(class_id)

        elif split == 'val':
            self.data_dir = os.path.join(root_dir, 'val')
            self.images = []
            self.labels = []

            annotations_file = os.path.join(self.data_dir, 'val_annotations.txt')
            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    class_id = parts[1]
                    self.images.append(os.path.join(self.data_dir, 'images', img_name))
                    self.labels.append(class_id)

        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        label_idx = self.class_to_idx[label]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label_idx

def get_tiny_imagenet_dataloaders(batch_size=16, data_dir=data_dir):
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TinyImageNetDataset(root_dir=data_dir, split='train', transform=transform_train)
    val_dataset = TinyImageNetDataset(root_dir=data_dir, split='val', transform=transform_test)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

train_loader, val_loader, test_loader, classes = get_tiny_imagenet_dataloaders()

__all__ = ['train_loader','val_loader','test_loader','classes']

