import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def visualize_nifti(nifti_file, num_slices=80, smaller_ratio=30, threshold=130):
    """
    Visualize non-empty slices of a NIfTI file, with a custom ratio of slices before and after the center.
    :param nifti_file: Path to the NIfTI file.
    :param num_slices: Total number of slices to display (default is 80).
    :param smaller_ratio: Number of slices before the center (default is 30).
    :param threshold: Minimum pixel value to consider a slice non-empty.
    """
    # Load the NIfTI image
    img = nib.load(nifti_file)
    data = img.get_fdata()
    total_slices = data.shape[2]
    print(f"Total slices: {total_slices}")
    
    # Ensure that the smaller_ratio and larger_ratio fit the total number of slices
    larger_ratio = num_slices - smaller_ratio

    # Find non-empty slices based on the threshold
    non_empty_slices = [i for i in range(data.shape[2]) if np.max(data[:, :, i]) > threshold]
    print(f"Non-empty slices: {len(non_empty_slices)}")

    # Determine the center slice
    center_slice = total_slices // 2

    # Determine start and end indices based on the ratio
    start_slice = max(center_slice - smaller_ratio, 0)
    end_slice = min(center_slice + larger_ratio, total_slices)

    # Indices for the slices you want to display
    slice_indices = list(range(start_slice, center_slice)) + list(range(center_slice, end_slice))
    print(f"Displaying slices from {start_slice} to {center_slice} and from {center_slice} to {end_slice}")

    # Ensure we only display the intended number of slices
    slice_indices = slice_indices[:num_slices]

    # Determine the layout for subplots
    cols = 5  # You want 5 columns
    rows = (num_slices + cols - 1) // cols  # Calculate rows needed

    # Create a plot with multiple subplots to visualize the slices
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))  # Adjust size based on rows

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Plot the selected slices
    for i, idx in enumerate(slice_indices):
        axes[i].imshow(data[:, :, idx], cmap="gray")
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')  # Hide axes

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("centered_slices_custom_ratio.png")
    plt.show()

if __name__ == "__main__":
    # Manually input the path to your NIfTI file here
    nifti_file2 = "D:/VascularData/data/nii/002_S_0413/Sagittal_3D_FLAIR/2017-06-21_13_23_38.0/I863060/I863060_Sagittal_3D_FLAIR_20170621132338_3_cleaned.nii.gz"

    # Number of slices to display (centered around the middle of the dataset)
    num_slices = 80  # Total number of slices to display
    smaller_ratio = 25  # 30 slices before the center, 50 after

    visualize_nifti(nifti_file2, num_slices, smaller_ratio)
