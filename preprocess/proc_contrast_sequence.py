import os
import random
import time
import numpy as np

from multiprocessing import Process
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensityRangePercentiles, ScaleIntensityRange, NormalizeIntensity

from util import load_string_list, load_nii_data, save_nii_data, cut_patch


def load_and_patch_transforms_series(img, tar_img_size):
    transforms = Compose(
        [
            # EnsureChannelFirst(),
            Resize(spatial_size=(tar_img_size[0], tar_img_size[1], tar_img_size[2]), mode='trilinear'),
            ScaleIntensityRangePercentiles(lower=1., upper=99.9, b_min=0.0, b_max=1.0, clip=True, relative=False, channel_wise=False),
        ]
    )
    return transforms(img)


def cut_and_save_one_volume(image_file, patch_size_list, patch_num, save_root, start_index, tar_img_size):
    image, affine = load_nii_data(image_file)
    image_s1, affine_s1 = load_nii_data(image_file.replace('t1n.nii.gz', 't1c.nii.gz'))
    image_s2, affine_s2 = load_nii_data(image_file.replace('t1n.nii.gz', 't2w.nii.gz'))
    image_s3, affine_s3 = load_nii_data(image_file.replace('t1n.nii.gz', 't2f.nii.gz'))

    ds_name = image_file.split('/')[-3]
    nii_id = image_file.split('/')[-1].split('.nii.gz')[0]
    patch_path_list = []
    if len(image.shape) == 4:
        return []
    else:
        images = [image]

    n = 0
    for image_index, image in enumerate(images):
        image = image.transpose((2, 1, 0))  # ->z,y,x
        image_s1 = image_s1.transpose((2, 1, 0))  # ->z,y,x
        image_s2 = image_s2.transpose((2, 1, 0))  # ->z,y,x
        image_s3 = image_s3.transpose((2, 1, 0))  # ->z,y,x

        _patch_num = patch_num

        for i in range(_patch_num):
            n += 1
            patch_size = random.choice(patch_size_list)
            image_patch, cut_size = cut_patch(image, patch_size)
            z1, z2, y1, y2, x1, x2 = cut_size
            image_s1_patch = image_s1[z1:z2, y1:y2, x1:x2]
            image_s2_patch = image_s2[z1:z2, y1:y2, x1:x2]
            image_s3_patch = image_s3[z1:z2, y1:y2, x1:x2]
            image_patch = image_patch.transpose((2, 1, 0))  # -> xyz
            image_s1_patch = image_s1_patch.transpose((2, 1, 0))  # -> xyz
            image_s2_patch = image_s2_patch.transpose((2, 1, 0))  # -> xyz
            image_s3_patch = image_s3_patch.transpose((2, 1, 0))  # -> xyz

            image_patch = load_and_patch_transforms_series(np.expand_dims(image_patch, 0), tar_img_size).numpy()[0, ...]
            image_s1_patch = load_and_patch_transforms_series(np.expand_dims(image_s1_patch, 0), tar_img_size).numpy()[0, ...]
            image_s2_patch = load_and_patch_transforms_series(np.expand_dims(image_s2_patch, 0), tar_img_size).numpy()[0, ...]
            image_s3_patch = load_and_patch_transforms_series(np.expand_dims(image_s3_patch, 0), tar_img_size).numpy()[0, ...]
            save_name = os.path.join(save_root, '%s_%s_%s_%d.t1n.nii.gz' % (ds_name, nii_id, image_index, start_index + n))
            patch_path_list.append(save_name)

            # # check
            # save_nii_data(save_name, image_patch, affine)
            # save_nii_data(save_name.replace('t1n.nii.gz', 't1c.nii.gz'), image_s1_patch, affine_s1)
            # save_nii_data(save_name.replace('t1n.nii.gz', 't2w.nii.gz'), image_s2_patch, affine_s2)
            # save_nii_data(save_name.replace('t1n.nii.gz', 't2f.nii.gz'), image_s3_patch, affine_s3)
            np.save(save_name.replace('.nii.gz', '.npy'), image_patch)
            np.save(save_name.replace('t1n.nii.gz', 't1c.nii.gz').replace('.nii.gz', '.npy'), image_s1_patch)
            np.save(save_name.replace('t1n.nii.gz', 't2w.nii.gz').replace('.nii.gz', '.npy'), image_s2_patch)
            np.save(save_name.replace('t1n.nii.gz', 't2f.nii.gz').replace('.nii.gz', '.npy'), image_s3_patch)

    return patch_path_list


def generate_3d_patch(
        data_list,
        save_root,
        patch_size_list,
        patch_num=16,
        start_index=0,
        tar_img_size=[]
):
    num = len(data_list)
    patch_list_all = []
    for i, path in enumerate(data_list):
        t1 = time.time()
        patch_list = cut_and_save_one_volume(path, patch_size_list, patch_num, save_root, start_index, tar_img_size)
        patch_list_all += patch_list
        t2 = time.time()
        dur = t2 - t1
        if i % 20 == 0:
            print(f"[{i+1}/{num}], {path}, time={dur:.2f}s")


def main(process_num=1):
    patch_num = 100
    tar_img_size = [128, 128, 32]
    patch_size_list = [(32, 128, 128)]
    start_index = 0

    """ base_path: your base path"""
    base_path = '/mnt/data/ssl/data/pretrain/'

    """ 
        'contrast.txt': *t1n.nii.gz
        brats23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00499-000/BraTS-GLI-00499-000-t1n.nii.gz,
        brats23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-01208-000/BraTS-GLI-01208-000-t1n.nii.gz,
        brats23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00448-000/BraTS-GLI-00448-000-t1n.nii.gz,
        ...
    """

    data_list_path = f'{base_path}/data_list/contrast.txt'
    save_root = f'{base_path}/data/patch_random_contrast'

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