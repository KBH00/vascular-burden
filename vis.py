import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    nifti_file2 = "D:/VascularData/data/nii/002_S_0413/Sagittal_3D_FLAIR/2017-06-21_13_23_38.0/I863060/I863060_Sagittal_3D_FLAIR_20170621132338_3_cleaned.nii.gz"

    num_slices = 80  # Total number of slices to display
    smaller_ratio = 25  # 30 slices before the center, 50 after

    visualize_nifti(nifti_file2, num_slices, smaller_ratio)
