import codecs
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def cut_patch(img_array, patch_size):
    img_shape = img_array.shape  # z,y,x
    size_z, size_y, size_x = patch_size
    center_x_min, center_x_max = size_x // 2, img_shape[2] - size_x // 2 - 1
    center_y_min, center_y_max = size_y // 2, img_shape[1] - size_y // 2 - 1
    center_z_min, center_z_max = size_z // 2, img_shape[0] - size_z // 2 - 1
    if center_x_min >= center_x_max:
        x1, x2 = 0, img_shape[2]
    else:
        center_x = random.randint(center_x_min, center_x_max)
        x1, x2 = center_x - size_x // 2, center_x + size_x // 2
    if center_y_min >= center_y_max:
        y1, y2 = 0, img_shape[1]
    else:
        center_y = random.randint(center_y_min, center_y_max)
        y1, y2 = center_y - size_y // 2, center_y + size_y // 2
    if center_z_min >= center_z_max:
        z1, z2 = 0, img_shape[0]
    else:
        center_z = random.randint(center_z_min, center_z_max)
        z1, z2 = center_z - size_z // 2, center_z + size_z // 2
    img_patch = img_array[z1:z2, y1:y2, x1:x2]
    return img_patch, [z1, z2, y1, y2, x1, x2]


def load_nii_data(nii_file):
    nii_data = nib.load(nii_file)
    return nii_data.get_fdata(), nii_data.affine


def save_nii_data(nii_file, data, affine=np.eye(4)):
    data_nii = nib.Nifti1Image(data, affine)
    nib.save(data_nii, nii_file)


def load_nii_data_sitk(nii_file):
    nii_data = sitk.GetArrayFromImage(sitk.ReadImage(nii_file))
    return nii_data, []


def save_nii_data_sitk(nii_file, data):
    out_image = sitk.GetImageFromArray(data)
    sitk.WriteImage(out_image, nii_file)


def load_string_list(file_path, is_utf8=False):
    if is_utf8:
        f = codecs.open(file_path, 'r', 'utf-8')
    else:
        f = open(file_path)
    l = []
    for item in f:
        item = item.strip()
        if len(item) == 0:
            continue
        l.append(item)
    f.close()
    return l


def save_string_list(file_path, l, is_utf8=False):
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, 'w')
    for item in l[:-1]:
        f.write(item + '\n')
    if len(l) >= 1:
        f.write(l[-1])
    f.close()