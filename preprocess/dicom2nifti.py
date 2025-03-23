import pydicom
import os
import numpy as np
import nibabel as nib


""" set data path """
parent_folder = '/mnt/data/ssl/data/RICORD/download/'
data_split_folder = '/mnt/data/ssl/data/RICORD/data_split/'
save_folder = '/mnt/data/ssl/data/RICORD/save/'


def path_to_original_name(path):
    parts = path.split('/')

    id_part = parts[-3]
    date_part = parts[-2].split('-')[0] + "-" + parts[-2].split('-')[1] + "-" + parts[-2].split('-')[2]
    scan_info = parts[-1]

    original_name_parts = [
        "RICORD_nii",
        parts[-4],
        "ID" + id_part.replace(parts[-4] + "-", "") + "_date" + date_part + "_" + scan_info.replace(" ", "-")
    ]
    original_name = "/".join(original_name_parts)
    original_name = original_name.replace(" ", "-") + ".nii.gz"
    
    return original_name


def find_deepest_subfolders(parent_folder):
    deepest_subfolders = []
    max_depth = 0
    for root, dirs, files in os.walk(parent_folder):
        current_depth = root.count(os.sep)
        if current_depth > max_depth:
            max_depth = current_depth
            deepest_subfolders = [root]
        elif current_depth == max_depth:
            deepest_subfolders.append(root)
    return sorted(deepest_subfolders)


def dicom_series_to_nifti(dicom_dir, output_file, suffix):
    output_path = os.path.join(suffix, output_file)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicom_files = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    print(dicom_dir)
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))

    dicom_slices = [file.pixel_array for file in dicom_files]
    ori_len = len(dicom_slices)
    dicom_slices = remove_outliers(dicom_slices)
    now_len = len(dicom_slices)
    if now_len != ori_len:
        print(f"{ori_len} -> {now_len}")
    image_data = np.stack(dicom_slices, axis=-1)

    affine = np.eye(4)
    nifti_image = nib.Nifti1Image(image_data, affine)
    nib.save(nifti_image, os.path.join(suffix, output_file))


def remove_outliers(dicom_slices):
    shape_list = [i.shape for i in dicom_slices]
    shape = max(set(shape_list), key=shape_list.count)
    new_dicom_slices = []
    for slice in dicom_slices:
        if slice.shape == shape:
            new_dicom_slices.append(slice)
    return new_dicom_slices


deepest_subfolders = find_deepest_subfolders(parent_folder)
print(f"Deepest Subfolders number: {len(deepest_subfolders)}")


name_list = []
for name_file in ["RICORD_train.txt", "RICORD_val.txt", "RICORD_test.txt"]:
    name_path = data_split_folder + name_file
    print(name_path)
    with open(name_path) as f:
        for line in f:
            name_list.append(line.split(" ")[0])


match_num = 0
total_num = len(name_list)
for i in deepest_subfolders:
    if len(os.listdir(i)) <= 10 or "11-01-2003-NA-NA-00585/2.000000-NA-03193" in i:
        continue
    file_path = path_to_original_name(i)
    if file_path in name_list:
        if '3.000000-NA-11240' in i:
            print(i)
        match_num += 1
        dicom_series_to_nifti(i, path_to_original_name(i), suffix=save_folder)
print(match_num, total_num)
