import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def visualize_volume(volumes, num_slices=5):
    B, _, H, W = volumes.shape  # Adjust to match the shape [batch size, 1, 128, 128]
    
    # Take the first volume from the batch (index 0)
    volume = volumes[0, 0].cpu().numpy()  # Shape: [128, 128]
    
    # Choose the step to evenly sample slices from the height (H)
    slice_step = max(1, H // num_slices)
    
    plt.imshow(volume, cmap="gray")  # No need for volume[slice_idx] since it's already 2D
    plt.axis('off')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_anomaly_map_with_original(original, anomaly_map, anomaly_score, threshold=0.5):
    batch_size = original.shape[0]  # Get the batch size
    # Convert tensors to NumPy arrays
    original = original.cpu().numpy()  # shape: [B, 1, H, W]
    anomaly_map = anomaly_map.cpu().numpy()  # shape: [B, 1, H, W]

    # Create a figure with multiple subplots for each image in the batch
    fig, axs = plt.subplots(batch_size, 2, figsize=(10, batch_size * 5))

    for i in range(batch_size):
        # Extract the current original and anomaly map
        original_img = original[i, 0]  # shape: [H, W]
        anomaly_img = anomaly_map[i, 0]  # shape: [H, W]

        # Create a mask for the anomaly map based on the threshold
        mask = anomaly_score[i] > threshold

        # Create a masked anomaly map
        masked_anomaly_map = np.zeros_like(anomaly_img)  # Initialize a zero array for the masked anomaly map
        if mask:  # If the anomaly score exceeds the threshold
            masked_anomaly_map = anomaly_img  # Keep the entire anomaly map
        else:
            masked_anomaly_map[:] = np.nan  # Set the entire masked map to NaN to visualize it as empty

        # Display the original image
        axs[i, 0].imshow(original_img, cmap='gray')
        axs[i, 0].set_title(f'Original Image {i+1}')
        axs[i, 0].axis('off')

        # Display the masked anomaly map
        axs[i, 1].imshow(masked_anomaly_map, cmap='hot')
        axs[i, 1].set_title(f'Anomaly Map {i+1} (Score: {anomaly_score[i].item():.4f})')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_volume(volumes, num_slices=5):
    """
    Visualize slices of a given 3D volume.

    Args:
        volumes (torch.Tensor): A batch of 3D volumes. Shape: (B, C, D, H, W).
        num_slices (int): Number of slices to visualize from the volume.
    """
    # Assume volumes is in shape (B, 1, D, H, W)
    B, D, H, W = volumes.shape
    print(B, D, H, W)
    # Visualize the slices of the first volume in the batch
    volume = volumes[0, 0].cpu().numpy()  # Shape: (D, H, W)
    
    # Choose the step to evenly sample slices
    slice_step = max(1, D // num_slices)

    plt.figure(figsize=(15, 5))
    for i in range(0, num_slices):
        slice_idx = i * slice_step
        plt.subplot(1, num_slices, i + 1)
        plt.imshow(volume[slice_idx], cmap="gray")
        plt.title(f"Slice {slice_idx}")
        plt.axis('off')
    plt.show()


def visualize_nifti(nifti_file, num_slices=80, smaller_ratio=30, threshold=130):
    """
    Visualize non-empty slices of a NIfTI file, with a custom ratio of slices before and after the center.
    :param nifti_file: Path to the NIfTI file.
    :param num_slices: Total number of slices to display (default is 80).
    :param smaller_ratio: Number of slices before the center (default is 30).
    :param threshold: Minimum pixel value to consider a slice non-empty.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    data = resize(data, [80, 128, 128])
    print(data.shape)
    total_slices = data.shape[2]
    print(f"Total slices: {total_slices}")
    
    larger_ratio = num_slices - smaller_ratio

    non_empty_slices = [i for i in range(data.shape[2]) if np.max(data[:, :, i]) > threshold]
    print(f"Non-empty slices: {len(non_empty_slices)}")

    center_slice = total_slices // 2

    start_slice = max(center_slice - smaller_ratio, 0)
    end_slice = min(center_slice + larger_ratio, total_slices)

    slice_indices = list(range(start_slice, center_slice)) + list(range(center_slice, end_slice))
    print(f"Displaying slices from {start_slice} to {center_slice} and from {center_slice} to {end_slice}")

    slice_indices = slice_indices[:num_slices]

    cols = 5 
    rows = (num_slices + cols - 1) // cols  

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3)) 

    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        axes[i].imshow(data[:, :, idx], cmap="gray")
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')  # Hide axes

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("centered_slices_custom_ratio.png")
    plt.show()
from skimage.transform import resize

if __name__ == "__main__":
    nifti_file2 = "D:/VascularData/data/nii/002_S_0413/Sagittal_3D_FLAIR/2017-06-21_13_23_38.0/I863060/I863060_Sagittal_3D_FLAIR_20170621132338_3_cleaned.nii.gz"

    num_slices = 80  # Total number of slices to display
    smaller_ratio = 25  # 30 slices before the center, 50 after

    visualize_nifti(nifti_file2, num_slices, smaller_ratio)
