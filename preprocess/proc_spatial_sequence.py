import os
import random
import time
import numpy as np

from multiprocessing import Process
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensityRangePercentiles, ScaleIntensityRange, NormalizeIntensity

from util import load_string_list, load_nii_data, load_nii_data_sitk, save_nii_data, cut_patch


def load_and_patch_transforms(img, tar_img_size):
    transforms = Compose(
        [
            # EnsureChannelFirst(),
            Resize(spatial_size=(tar_img_size[0], tar_img_size[1], tar_img_size[2]), mode='trilinear'),
            ScaleIntensityRangePercentiles(lower=1., upper=99.9, b_min=0.0, b_max=1.0, clip=True, relative=False, channel_wise=False),
        ]
    )
    return transforms(img)


def cut_and_save_one_volume(image_file, patch_size_list, patch_num, save_root, start_index, tar_img_size):
    if image_file.endswith('.mha'):
        image, affine = load_nii_data_sitk(image_file)
        image = image.transpose((2, 1, 0))
    else:
        image, affine = load_nii_data(image_file)

    ds_name = image_file.split('/')[-3]
    nii_id = image_file.split('/')[-1].split('.nii.gz')[0].split('.mha')[0]

    patch_path_list = []
    if len(image.shape) == 4:
        return []
    else:
        images = [image]

    n = 0
    for image_index, image in enumerate(images):
        image = image.transpose((2, 1, 0))  # ->z,y,x
        if image.shape[0] < patch_size_list[0][2]:
            _patch_num = int(patch_num / 1.5)
        else:
            _patch_num = patch_num

        for i in range(_patch_num):
            n += 1
            patch_size = random.choice(patch_size_list)
            image_patch, cut_size = cut_patch(image, patch_size)
            image_patch = image_patch.transpose((2, 1, 0))  # -> xyz
            image_patch = load_and_patch_transforms(np.expand_dims(image_patch, 0), tar_img_size).numpy()[0, ...]
            save_name = os.path.join(save_root, '%s_%s_%s_%d.nii.gz' % (ds_name, nii_id, image_index, start_index + n))
            patch_path_list.append(save_name)

            # # check
            # save_nii_data(save_name, image_patch, affine)
            np.save(save_name.replace('.nii.gz', '.npy'), image_patch)
    return patch_path_list


def generate_3d_patch(
        data_list,
        save_root,
        patch_size_list,
        patch_num=16,
        start_index=0,
        tar_img_size=[],
):

    # save 3d patch
    num = len(data_list)
    patch_list_all = []
    for i, path in enumerate(data_list):
        # if 'LUNA16' in path:
            # continue
        t1 = time.time()
        patch_list = cut_and_save_one_volume(path, patch_size_list, patch_num, save_root, start_index, tar_img_size)
        patch_list_all += patch_list
        t2 = time.time()
        dur = t2 - t1
        if i % 20 == 0:
            print(f"[{i+1}/{num}], {path}, time={dur:.2f}s")


def main(process_num=1):
    patch_num = 50
    tar_img_size = [128, 128, 128]
    patch_size_list = [(128, 128, 128)]
    start_index = 0

    """ base_path: your base path"""
    base_path = '/mnt/data/ssl/data/pretrain/'

    """ 
        'spatial.txt': concatenate all the dataset
        TCIA_Covid19/CTImagesInCOVID19/volume-covid19-A-0706_day000.nii.gz,
        TCIA_Covid19/CTImagesInCOVID19/volume-covid19-A-0715_day012.nii.gz,
        TCIA_Covid19/CTImagesInCOVID19/volume-covid19-A-0705_day004.nii.gz,
        ...
    """

    data_list_path = f'{base_path}/data_list/spatial.txt'
    save_root = f'{base_path}/data/patch_random_spatial'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data_list = load_string_list(data_list_path)
    num = len(data_list)
    print('num:', num)

    if process_num <= 1:
        generate_3d_patch(data_list, save_root, patch_size_list, patch_num, start_index, tar_img_size)
    else:
        process_list = []
        stride = num // process_num + 1
        for i in range(process_num):
            start = i * stride
            end = min(stride * (i + 1) - 1, num)
            file_list = data_list[start: end + 1]
            p = Process(target=generate_3d_patch,
                        args=(file_list, save_root, patch_size_list, patch_num, start_index, tar_img_size))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()


if __name__ == "__main__":
    process_num = 1
    main(process_num)
