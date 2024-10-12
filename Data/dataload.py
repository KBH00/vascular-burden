import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class Nifti3DDataset(Dataset):
    def __init__(self, directories, transform=None, labels=None, num_slices=80, smaller_ratio=30):
        """
        Custom dataset for loading 3D NIfTI images with a center-based 30-50 slice cut.

        Args:
            directories (list): List of directories containing .nii files.
            transform (callable, optional): Optional transform to be applied on a sample.
            labels (list, optional): Optional labels for supervised learning.
            num_slices (int, optional): Total number of slices to extract (default is 80).
            smaller_ratio (int, optional): Number of slices before the center (default is 30).
        """
        self.directories = directories
        self.transform = transform
        self.labels = labels
        self.num_slices = num_slices
        self.smaller_ratio = smaller_ratio
        self.larger_ratio = num_slices - smaller_ratio

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        nii_file_path = self.directories[idx]

        if not os.path.exists(nii_file_path):
            raise ValueError(f"No NIfTI file found in directory: {nii_file_path}")

        nifti_data = nib.load(nii_file_path)
        volume = nifti_data.get_fdata().astype(np.float32)

        total_slices = volume.shape[2]
        center_slice = total_slices // 2

        start_slice = max(center_slice - self.smaller_ratio, 0)
        end_slice = min(center_slice + self.larger_ratio, total_slices)

        volume_cropped = volume[:, :, start_slice:end_slice]

        volume_resized = F.interpolate(torch.tensor(volume_cropped).unsqueeze(0).unsqueeze(0), size=(self.num_slices, 128, 128), mode='trilinear', align_corners=False)
        volume_resized = volume_resized.squeeze(0).float()

        if self.transform:
            volume_resized = self.transform(volume_resized)

        label = self.labels[idx] if self.labels is not None else -1
        return volume_resized, label


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


def get_dataloaders(train_base_dir, modality, batch_size=4, transform=None, validation_split=0.1, test_split=0.1, seed=42):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        train_base_dir (str): Base directory containing training NIfTI directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data.
        validation_split (float, optional): Fraction of the dataset to use for validation.
        test_split (float, optional): Fraction of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """

    #original
        #     transform = transforms.Compose([
        #     transforms.Normalize((0.5,), (0.5,)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomRotation(30),
        #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # ])
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


# Example usage
# if __name__ == "__main__":
#     path = "D:/VascularData/data/nii"
#     get_dataloaders(path, "FLAIR")
