import os
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from torch.utils.data import random_split
from torchio import RandomAffine, RandomElasticDeformation, RandomNoise, RandomMotion, RandomBiasField, RandomFlip, RandomBlur, RescaleIntensity, RandomGamma, Compose


class Nifti3DDataset:
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

        if self.transform:
            volume_cropped = self.transform(volume_cropped)

        label = self.labels[idx] if self.labels is not None else -1

        # Wrap the data into a TorchIO Subject
        subject = tio.Subject(
            image=tio.Image(tensor=torch.tensor(volume_cropped).unsqueeze(0), type=tio.INTENSITY),
            label=label
        )

        return subject


def find_nii_directories(base_dir, modality="FLAIR"):
    """
    Recursively find all directories containing .nii.gz files with specific criteria.
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
    """

    if transform is None:
        transform = Compose([
            RescaleIntensity(out_min_max=(0, 1)),  # Normalize intensity
            RandomAffine(scales=(0.9, 1.1), degrees=15),  # Random scaling and rotation
            RandomElasticDeformation(num_control_points=5, max_displacement=7.5),  # Non-linear elastic deformation
            RandomNoise(mean=0.0, std=0.05),  # Add Gaussian noise
            RandomMotion(degrees=10, translation=10),  # Simulate motion artifacts
            RandomBiasField(coefficients=0.5),  # Simulate intensity inhomogeneity
            RandomFlip(axes=(0, 1, 2)),  # Random flipping in 3D axes
            RandomBlur(std=(0, 2)),  # Random blurring
            RandomGamma(log_gamma=(0.5, 1.5)),  # Adjust gamma to simulate contrast changes
        ])
    torch.manual_seed(seed)

    train_directories = find_nii_directories(base_dir=train_base_dir, modality=modality)
    train_dataset = Nifti3DDataset(train_directories, transform=transform)

    total_size = len(train_dataset)
    validation_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)
    train_size = total_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])

    # Use SubjectsLoader from TorchIO instead of PyTorch DataLoader
    train_loader = tio.SubjectsDataset(train_dataset)
    validation_loader = tio.SubjectsDataset(validation_dataset)
    test_loader = tio.SubjectsDataset(test_dataset)

    return train_loader, validation_loader, test_loader


# Example usage
# if __name__ == "__main__":
#     path = "D:/VascularData/data/nii"
#     get_dataloaders(path, "FLAIR")
