import os
import sys
import argparse
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pydicom
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from skimage.transform import resize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.models import FeatureReconstructor
from utils.pytorch_ssim import SSIMLoss
from models.feature_extractor import Extractor
from nilearn.masking import compute_brain_mask

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # If tqdm is not installed, use a dummy loop


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
        return volume_resized, nii_file_path


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
        print(root)
        for file in files:
            if file.endswith(".nii.gz") and modality in root and "cleaned" in file:
                nii_directories.append(os.path.join(root, file))
                break
    return nii_directories

def visualize_anomaly_overlay(original_slice: torch.Tensor, anomaly_map: torch.Tensor, save_path: str = None):
    """
    Visualize the overlay of the anomaly map on the original MR image.

    Args:
        original_slice (torch.Tensor): The original MR image slice of shape (H, W).
        anomaly_map (torch.Tensor): The anomaly map of shape (H, W).
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
    """

    original_slice = original_slice.squeeze().cpu().numpy()
    anomaly_map = anomaly_map.squeeze().cpu().numpy()

    plt.figure(figsize=(6, 6))
    
    plt.imshow(original_slice, cmap='gray', interpolation='none')
    plt.imshow(anomaly_map, cmap='hot', alpha=0.5, interpolation='none')
    
    plt.colorbar()
    plt.title('MR Image with Anomaly Map Overlay')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Anomaly map overlay saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inference with Feature Autoencoder')
    parser.add_argument('--checkpoint', type=str, default="./saved_models/epoch_6_batchsize_4_lr_0.001.pth", help='Path to the model checkpoint file')
    parser.add_argument('--input_dir', type=str, default="D:/VascularData/data/nii/", help='Directory containing DICOM directories for inference')
    parser.add_argument('--output_dir', type=str, default='./anomaly_maps', help='Directory to save anomaly map visualizations')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.hidden_dims = [100, 150, 200, 300]
    config.generator_hidden_dims = [300, 200, 150, 100]
    config.discriminator_hidden_dims = [100, 150, 200, 300]
    config.dropout = 0.2
    config.extractor_cnn_layers = ['layer1', 'layer2']
    config.keep_feature_prop = 1.0
    config.random_extractor = False
    config.loss_fn = 'ssim'

    model = FeatureReconstructor(config).to(args.device)
    
    model.load(args.checkpoint)
    model.eval()

    nii_directories = find_nii_directories(args.input_dir)
    if len(nii_directories) == 0:
        print(f"No DICOM directories found in {args.input_dir}")
        return

    inference_dataset = Nifti3DDataset(nii_directories)
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch_idx, (volumes, directories) in enumerate(tqdm(inference_loader, desc="Inference")):
            volumes = volumes.to(args.device)  
            B, C, D, H, W = volumes.shape
            print(volumes.shape)
            # Flatten the slices into (B*D, 1, H, W) for per-slice anomaly detection
            volumes_slices = volumes.view(B * D, C, H, W)  # Shape: (B*D, 1, H, W)

            anomaly_map, anomaly_score = model.predict_anomaly(volumes_slices)  # anomaly_map: (B*D, 1, H, W)
            anomaly_map = anomaly_map.view(B, D, 1, H, W)  # Reshape back to (B, D, 1, H, W)

            middle_slice = D // 2
            for i in range(B):
                dir_path = directories[i]
                volume_id = os.path.basename(dir_path.rstrip('/\\'))
                path_id = dir_path.split('/')[-1].replace('\\','.')

                original_slice = volumes[i, 0, middle_slice, :, :]  # Shape: (H, W)
                single_anomaly_map = anomaly_map[i, middle_slice, 0, :, :]  # Shape: (H, W)

                save_path = os.path.join(args.output_dir, f"{path_id}.png")
                visualize_anomaly_overlay(original_slice, single_anomaly_map, save_path=save_path)

                print(f"Volume: {volume_id}, Anomaly Score: {anomaly_score[i].item():.4f}")

    print("Inference completed.")


if __name__ == '__main__':
    main()
