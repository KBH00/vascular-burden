import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from functools import partial
from Data.data_utils import load_files_to_ram, load_nii_nn


class Nifti3DDataset(Dataset):
    def __init__(self, directories, transform=None, labels=None, config=None):
        """
        Custom dataset for loading 3D NIfTI images using `load_nii_nn`.

        Args:
            directories (list): List of directories containing .nii files.
            transform (callable, optional): Optional transform to be applied on a sample.
            labels (list, optional): Optional labels for supervised learning.
            config (Namespace, optional): Configuration containing loading parameters.
        """
        self.directories = directories
        self.transform = transform
        self.labels = labels
        self.config = config

        load_fn = partial(load_nii_nn,
                          slice_range=config.slice_range,
                          size=config.image_size,
                          normalize=config.normalize,
                          equalize_histogram=config.equalize_histogram)

        self.volumes = load_files_to_ram(self.directories, load_fn)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume_resized = self.volumes[idx]

        if self.transform:
            volume_resized = self.transform(volume_resized)

        label = self.labels[idx] if self.labels is not None else -1
        return volume_resized, label

class TrainDataset(Dataset):
    """
    Training dataset. No anomalies, no segmentation maps.
    """

    def __init__(self, imgs: np.ndarray):
        """
        Args:
            imgs (np.ndarray): Training slices
        """
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

def find_nii_directories(base_dir, modality="FLAIR"):
    """
    Recursively find all directories containing .nii.gz files with specific criteria.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .nii.gz file with the modality.
    """
    nii_directories = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nii.gz") and modality in file and "cleaned" in file:
                nii_directories.append(os.path.join(root, file))
                break
    return nii_directories

from typing import List, Tuple, Sequence

def load_images(files: List[str], config) -> np.ndarray:
    """Load images from a list of files.
    Args:
        files (List[str]): List of files
        config (Namespace): Configuration
    Returns:
        images (np.ndarray): Numpy array of images
    """
    load_fn = partial(load_nii_nn,
                      slice_range=config.slice_range,
                      size=config.image_size,
                      normalize=config.normalize,
                      equalize_histogram=config.equalize_histogram)
    return load_files_to_ram(files, load_fn)

def get_dataloaders(train_base_dir, modality, batch_size=4, transform=None,
                    validation_split=0.1, test_split=0.1, seed=42, config=None):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        train_base_dir (str): Base directory containing training NIfTI directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data.
        validation_split (float, optional): Fraction of the dataset to use for validation.
        test_split (float, optional): Fraction of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility.
        config (Namespace, optional): Configuration for loading parameters.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    torch.manual_seed(seed)

    print("Data load....")

    train_directories = find_nii_directories(base_dir=train_base_dir, modality=modality)
    train_directories = train_directories[:4]
    train_imgs = np.concatenate(load_images(train_directories, config))
    #train_dataset = Nifti3DDataset(train_directories, transform=transform, config=config)
    train_dataset =TrainDataset(train_imgs)

    total_size = len(train_dataset)
    validation_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)
    train_size = total_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(validation_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, validation_loader, test_loader
