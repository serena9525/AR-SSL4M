import numpy as np
from monai import transforms


def resize_npz(src_file, tar_file, tar_shape):
    # monai
    transforms_trilinear = transforms.Resize(spatial_size=tar_shape, mode=("trilinear"))

    items = np.load(src_file)
    print(items.files)

    save_dict = {}
    for key in items.files:
        value = items[key]

        if 'labels' in key:
            save_dict[key] = value
        else:
            new_value = np.zeros((value.shape[0], tar_shape[0], tar_shape[1], tar_shape[2])).astype(value.dtype)
            for i in range(value.shape[0]):
                new_value[i, ...] = transforms_trilinear(value[i:i + 1, ...])[0]
            save_dict[key] = new_value

    np.savez(tar_file, **save_dict)


if __name__ == '__main__':
    src_file = '/mnt/data/ssl/data/nodule/nodulemnist3d.npz'
    tar_file = '/mnt/data/ssl/data/nodule_128/nodulemnist3d.npz'
    tar_shape = [128, 128, 128]

    resize_npz(src_file, tar_file, tar_shape)
