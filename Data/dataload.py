import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage.transform import resize
from nilearn.masking import compute_brain_mask

class Nifti3DDataset(Dataset):
    def __init__(self, directories, transform=None, labels=None):
        """
        Custom dataset for loading 3D NIfTI images.

        Args:
            directories (list): List of directories containing .nii files.
            transform (callable, optional): Optional transform to be applied on a sample.
            labels (list, optional): Optional labels for supervised learning.
        """
        self.directories = directories
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        directory = self.directories[idx]
        nii_file_path = os.path.join(directory, "converted.nii")

        if not os.path.exists(nii_file_path):
            raise ValueError(f"No NIfTI file found in directory: {directory}")

        nifti_data = nib.load(nii_file_path)
        volume = nifti_data.get_fdata().astype(np.float32)

        mask = compute_brain_mask(nifti_data)
        volume = volume * mask.get_fdata()

        volume_resized = F.interpolate(torch.tensor(volume).unsqueeze(0).unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False)
        volume_resized = volume_resized.squeeze(0)
        volume_resized = volume_resized.float()  

        if self.transform:
            volume_resized = self.transform(volume_resized)

        label = self.labels[idx] if self.labels is not None else -1
        return volume_resized, label

def find_dcm_directories(base_dir):
    """
    Recursively find all directories containing .dcm files.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .dcm file.
    """
    dcm_directories = []
    for root, dirs, files in os.walk(base_dir):
        if any(file.endswith(".dcm") for file in files):
            dcm_directories.append(root)
    return dcm_directories
 
def find_nii_directories(base_dir, modality="FLAIR"):
    """
    Recursively find all directories containing .nii files.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .nii file.
    """
    nii_directories = []
    for root, dirs, files in os.walk(base_dir):
        if any(file == "converted.nii" for file in files):
            if root.count(modality) == 1:
                nii_directories.append(root)
            #else statement needed <<<!!!!

    return nii_directories

def get_dataloaders(train_base_dir, modality, batch_size=4, transform=None, validation_split=0.1, test_split=0.1, seed=42):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        train_base_dir (str): Base directory containing training NIfTI directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data. Default is Normalize((0.5,), (0.5,)).
        validation_split (float, optional): Fraction of the dataset to use for validation. Default is 0.2.
        test_split (float, optional): Fraction of the dataset to use for testing. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    torch.manual_seed(seed)

    print("Data load....")

    train_directories = find_nii_directories(base_dir=train_base_dir, modality=modality)
    train_dataset = Nifti3DDataset(train_directories, transform=transform)

    total_size = len(train_dataset)
    validation_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)
    train_size = total_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Data Loader ready...")

    return train_loader, validation_loader, test_loader

# if __name__ == "__main__":
#     path = "D:/Data/FLAIR_T2_ss/ADNI"
#     train_directories = find_nii_directories(path)
#     print(train_directories)
