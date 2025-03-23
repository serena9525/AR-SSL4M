import numpy as np
import torchio as tio


class Transform3D:

    def __init__(self, train_augmentation=False, mul=None, target_shape=None, data_flag=''):
        self.train_transform = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
        self.train_augmentation = train_augmentation

    def __call__(self, voxel):
        voxel = voxel.transpose(0, 3, 2, 1)

        if self.train_augmentation:
            voxel = self.train_transform(voxel)

        return voxel.astype(np.float32)
