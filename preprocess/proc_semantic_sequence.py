import csv
import os
import numpy as np
import nibabel as nib

from glob import glob
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensityRangePercentiles, ScaleIntensityRange, NormalizeIntensity
from util import load_nii_data, save_nii_data


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def load_and_patch_transforms(img, tar_img_size):
    transforms = Compose(
        [
            # EnsureChannelFirst(),
            Resize(spatial_size=(tar_img_size[0], tar_img_size[1], tar_img_size[2]), mode='trilinear'),
            ScaleIntensityRangePercentiles(lower=1., upper=99.9, b_min=0.0, b_max=1.0, clip=True, relative=False, channel_wise=False),
        ]
    )
    return transforms(img)


def main():
    tar_img_size = [128, 128, 32]

    """ base_path: your base path"""
    base_path = '/mnt/data/ssl/data/pretrain/'

    """ 
    DL_path: your DeepLesion dataset path, you need to use DL_save_nifti.py to obtain Images_nifti first 
    """
    DL_path = '/mnt/data/ssl/data/DeepLesion/'

    data_path = f'{DL_path}/Images_nifti'
    anno_path = f'{DL_path}/DL_info.csv'
    save_root = f'{base_path}/data/patch_random_semantic'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    annos = readCSV(anno_path)
    pi_set, pssid_set = set(), set()

    for index in range(1, len(annos)):
        anno = annos[index]
        if anno[9] == '-1':
            continue

        pssid_set.add('_'.join(anno[0].split('_')[:3]))
        pi_set.add(anno[0].split('_')[0])
        image_name = '_'.join(anno[0].split('_')[:3])
        bbox = float(anno[6].split(',')[0]), float(anno[6].split(',')[1][1:]), float(anno[6].split(',')[2][1:]), float(anno[6].split(',')[3][1:])
        x_center, y_center = int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)
        slice_range = anno[11].split(',')[0], anno[11].split(',')[-1].split(' ')[-1]

        nii_name = glob(f'{data_path}/{image_name}_*{slice_range[0]}-*{slice_range[1]}.nii.gz')[0]
        nii_data, affine = load_nii_data(nii_name)
        x_shape, y_shape = nii_data.shape[0], nii_data.shape[1]
        image_patch = nii_data[max(0, y_center - 64): min(y_center + 64, x_shape), max(0, x_center - 64): min(x_center + 64, y_shape), :]
        image_patch = load_and_patch_transforms(np.expand_dims(image_patch, 0), tar_img_size).numpy()[0, ...]

        save_name = os.path.join(save_root, f'{image_name}_{index}_{anno[9]}.nii.gz')

        # # check
        # save_nii_data(save_name, image_patch, affine)
        np.save(save_name.replace('.nii.gz', '.npy'), image_patch)


if __name__ == "__main__":
    main()
