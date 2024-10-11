import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
import SimpleITK as sitk
import cv2
import os
import nibabel as nib
from deepbrain import Extractor  # To use DeepBrain for brain mask extraction

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
    def fn(SLICE):
        plt.figure(figsize=(7,7))
        plt.imshow(arr[SLICE, :, :], cmap=cmap)
    interact(fn, SLICE=(0, arr.shape[0]-1))

def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
    assert arr.shape == mask.shape
    _arr = rescale_linear(arr, 0, 1)
    _mask = rescale_linear(mask, 0, 1).astype(np.uint8)

    def fn(SLICE):
        arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 1, 0), thickness)
        plt.figure(figsize=(7,7))
        plt.imshow(arr_with_contours)

    interact(fn, SLICE=(0, arr.shape[0]-1))

def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def add_suffix_to_filename(filename: str, suffix:str) -> str:
    if filename.endswith('.nii'):
        return filename.replace('.nii', f'_{suffix}.nii')
    elif filename.endswith('.nii.gz'):
        return filename.replace('.nii.gz', f'_{suffix}.nii.gz')
    else:
        raise RuntimeError('filename with unknown extension')

# Use SimpleITK for image reading and saving
BASE_DIR = "D:/Data/FLAIR_T2_ss/ADNI"

def find_nii_directories(base_dir, modality="FLAIR"):
    nii_directories = []
    for root, dirs, files in os.walk(base_dir):
        if any(file == "converted.nii" for file in files):
            if root.count(modality) == 1:
                nii_directories.append(root)
    return nii_directories

nii_path_s = find_nii_directories(base_dir=BASE_DIR)
for raw_dir_path in nii_path_s:
    raw_img_path = os.path.join(raw_dir_path, "converted.nii")
    raw_example = raw_img_path.split('\\')[-1]
    
    # Load the image using SimpleITK
    raw_img_sitk = sitk.ReadImage(raw_img_path)
    raw_img_arr = sitk.GetArrayFromImage(raw_img_sitk)  # Convert to numpy array

    # Display the 3D array
    print(f'shape = {raw_img_arr.shape} -> (Z, X, Y)')
    explore_3D_array(arr=raw_img_arr, cmap='nipy_spectral')

    # Use DeepBrain for brain mask extraction
    ext = Extractor()
    prob = ext.run(raw_img_arr)
    brain_mask = prob > 0.5  # Threshold for binary mask

    # Save the brain mask
    brain_mask_img = nib.Nifti1Image(brain_mask.astype(np.float32), affine=np.eye(4))
    out_filename = add_suffix_to_filename(raw_example, 'brainMaskByDL')
    nib.save(brain_mask_img, os.path.join(BASE_DIR, out_filename))

    # Show original image with mask contour
    explore_3D_array_with_mask_contour(raw_img_arr, brain_mask)

    # Mask the original image with the brain mask
    masked_img_arr = raw_img_arr * brain_mask

    # Display the masked image
    explore_3D_array(masked_img_arr)

    # Save the masked image
    masked_img = nib.Nifti1Image(masked_img_arr.astype(np.float32), affine=np.eye(4))
    out_filename_masked = add_suffix_to_filename(raw_example, 'brainMaskedByDL')
    nib.save(masked_img, os.path.join(BASE_DIR, out_filename_masked))

    print(f"Masked image saved: {out_filename_masked}")
