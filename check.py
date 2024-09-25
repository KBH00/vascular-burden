import pydicom
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

data_path = "D:/Download/FLAIR_T2_dataset/ADNI/"
folder_path = "D:/Download/FLAIR_T2_dataset/ADNI/003_S_4441/Sagittal_3D_FLAIR__MSV22_/2024-04-23_11_32_46.0/I10454354"
dicom_file = "/ADNI_003_S_4441_MR_Sagittal_3D_FLAIR_(MSV22)__br_raw_20240509082639271_54.dcm"
ds = pydicom.dcmread(folder_path+dicom_file)

def visualize_2D(ds):
    pixel_array = ds.pixel_array

    plt.imshow(pixel_array, cmap="gray")
    plt.title("DICOM Slice")
    plt.show()

def metadata(ds):
    if 'PixelSpacing' in ds:
        pixel_spacing = [float(value) for value in ds.PixelSpacing]  # Convert to floats
        print(f"Pixel Spacing (Row, Column): {pixel_spacing}")
    else:
        print("Pixel Spacing information not found in the DICOM file.")

    if 'SliceThickness' in ds:
        slice_thickness = float(ds.SliceThickness)  # Convert to float
        print(f"Slice Thickness: {slice_thickness}")
    else:
        print("Slice Thickness information not found in the DICOM file.")

    if 'PixelSpacing' in ds and 'SliceThickness' in ds:
        voxel_size = [pixel_spacing[0], pixel_spacing[1], slice_thickness]
        print(f"Voxel Size (mm): {voxel_size}")

def load_dicom_folder(folder_path):
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_files.sort()  # Ensure correct slice order
    slices = [pydicom.dcmread(f) for f in dicom_files]
    # Stack slices to create a 3D array
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    return volume, slices[0]

# import time
# def print_dimension_entire(data_dir):
#     image_counter = 0
#     for root, dirs, files in tqdm(os.walk(data_dir)):
#         if len(files) > 1:
#             volume, sample_slice = load_dicom_folder(root)
#             print_dimension(root, volume, sample_slice)

def get_dimension(volume, sample_slice):
    rows = sample_slice.Rows
    cols = sample_slice.Columns
    num_slices = volume.shape[-1]  
    return (rows, cols, num_slices)

from collections import defaultdict
def print_dimension_cnt(data_dir):
    dimension_count = defaultdict(int)  # Dictionary to count unique dimensions

    # Walk through all directories in the dataset
    for root, dirs, files in tqdm(os.walk(data_dir)):
        if len(files) > 1:
            volume, sample_slice = load_dicom_folder(root)
            dimensions = get_dimension(volume, sample_slice)
            dimension_count[dimensions] += 1
            print(f"Directory path {root} : {dimensions[0]} x {dimensions[1]} x {dimensions[2]} (Rows x Columns x Slices)")

    # Print the summary of unique dimension styles
    print("\nSummary of Dimension Styles:")
    for dim, count in dimension_count.items():
        print(f"{dim[0]} x {dim[1]} x {dim[2]} : {count} times")

# def print_dimension(root, volume, sample_slice):
#     rows = sample_slice.Rows
#     cols = sample_slice.Columns
#     num_slices = volume.shape[-1]  
#     print(f"Directory path {root} : {rows} x {cols} x {num_slices} (Rows x Columns x Slices)")

#     # plt.imshow(volume[:, :, volume.shape[2] // 2], cmap="gray")
#     # plt.title("Middle Slice of 3D Volume")
#     # plt.show()

def visual_2D_3D(volume, sample_slice):
    # Print dimension
    rows = sample_slice.Rows
    cols = sample_slice.Columns
    num_slices = volume.shape[-1]  
    print(f"3D Volume Dimensions: {rows} x {cols} x {num_slices} (Rows x Columns x Slices)")
    plt.imshow(volume[:, :, volume.shape[2] // 2], cmap="gray")
    plt.title("Middle Slice of 3D Volume")
    plt.show()

    # Display several slices from the 3D volume
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(volume[:, :, volume.shape[2] // 4], cmap="gray")
    ax[0].set_title('Slice 1/4')
    ax[1].imshow(volume[:, :, volume.shape[2] // 2], cmap="gray")
    ax[1].set_title('Slice 1/2')
    ax[2].imshow(volume[:, :, 3 * volume.shape[2] // 4], cmap="gray")
    ax[2].set_title('Slice 3/4')
    plt.show()


from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_3D(volume):
    # Perform marching cubes to extract isosurface
    verts, faces, _, _ = measure.marching_cubes(volume, level=0)

    # Visualize the 3D volume
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    ax.add_collection3d(mesh)

    # Set the view angles
    ax.view_init(30, 30)
    plt.show()

#volume, sample_slice = load_dicom_folder(folder_path)
data_dir = "D:/Download/FLAIR_T2_dataset/ADNI"
print_dimension_cnt(data_dir)



