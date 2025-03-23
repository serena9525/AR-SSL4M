import os
import torch
import h5py
import random
import math
import pickle
import collections
import numpy as np
import os.path as osp

from collections import Counter
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
from torch.utils import data
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform


def read_txt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = [f.strip() for f in data]
    return data


# copy from https://github.com/yulequan/UA-MT/blob/master/code/dataloaders/la_heart.py
class LAseg_Dataset(data.Dataset):
    def __init__(self, root,  max_iters=None, crop_size=(64, 64, 64), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255, ratio_labels=1, split="train"):

        self.root = root
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ratio_labels = ratio_labels
        self.split = split

        dataset_split = {'train': 'train.list', 'valid': 'test.list'}[split]

        self.img_list = read_txt(f'{root}/{dataset_split}')
        self.img_ids = [f'{root}/2018LA_Seg_Training_Set/{x}/mri_norm2.h5' for x in self.img_list]

        self.dataset_len = len(self.img_ids)
        if split == 'train':
            if self.ratio_labels < 1:
                self.img_ids = self.img_ids[:int(ratio_labels*self.dataset_len)]
        print("full number: {}, ratio: {}, now number: {}".format(self.dataset_len, self.ratio_labels, len(self.img_ids)))
        print("Start preprocessing....")
        if split == 'train':
            if not max_iters == None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.img_ids)
    
    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        if len(target_size) == 3:
            rows_missing = math.ceil(target_size[0] - img.shape[0])
            cols_missing = math.ceil(target_size[1] - img.shape[1])
            dept_missing = math.ceil(target_size[2] - img.shape[2])
            if rows_missing < 0:
                rows_missing = 0
            if cols_missing < 0:
                cols_missing = 0
            if dept_missing < 0:
                dept_missing = 0

            padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        else:
            rows_missing = math.ceil(target_size[1] - img.shape[1])
            cols_missing = math.ceil(target_size[2] - img.shape[2])
            dept_missing = math.ceil(target_size[3] - img.shape[3])
            if rows_missing < 0:
                rows_missing = 0
            if cols_missing < 0:
                cols_missing = 0
            if dept_missing < 0:
                dept_missing = 0

            padded_img = np.pad(img, ((0,0), (0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def locate_bbx(self, label):

        img_d, img_h, img_w = label.shape
        d0 = random.randint(0, img_d - self.crop_d)
        h0 = random.randint(0, img_h - self.crop_h)
        w0 = random.randint(0, img_w - self.crop_w)
        d1 = d0 + self.crop_d
        h1 = h0 + self.crop_h
        w1 = w0 + self.crop_w

        d0 = np.max([d0, 0])
        d1 = np.min([d1, img_d])
        h0 = np.max([h0, 0])
        h1 = np.min([h1, img_h])
        w0 = np.max([w0, 0])
        w1 = np.min([w1, img_w])

        return [d0, d1, h0, h1, w0, w1]
    
    def id2trainId(self, label):

        target = (label == 1)
        shape = label.shape

        results_map = np.zeros((1, shape[0], shape[1], shape[2])).astype(np.float32)

        results_map[0, :, :, :] = np.where(target, 1, 0)

        return results_map
    
    def __getitem__(self, index):
        image_name = self.img_ids[index]
        h5f = h5py.File(image_name, 'r')
        # image = h5f['image'][:].transpose(2,0,1)[np.newaxis]
        # label = h5f['label'][:].transpose(2,0,1)
        image = h5f['image'][:][np.newaxis]
        label = h5f['label'][:]

        if self.split == "train":

            image = self.pad_image(image, [1, self.crop_d, self.crop_h, self.crop_w])
            label = self.pad_image(label, [self.crop_d, self.crop_h, self.crop_w])

            [d0, d1, h0, h1, w0, w1] = self.locate_bbx(label)

            image = image[:, d0: d1, h0: h1, w0: w1]
            label = label[d0: d1, h0: h1, w0: w1]

        label = self.id2trainId(label)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy()


def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="image", label_key="label")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    # name = np.stack(name, 0)
    data_dict = {'image': image, 'label': label}
    tr_transforms = get_train_transform(patch_size=label.shape[2:])
    data_dict = tr_transforms(**data_dict)
    return data_dict


def get_train_loader(args, train_dataset, logger, batch_size, num_workers, distributed, world_size, collate_fn=my_collate, drop_last=True):

    train_sampler = None
    is_shuffle = True
    batch_size = args.batch_size

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # attributes = [attr for attr in dir(train_sampler) if not callable(getattr(train_sampler, attr))]
        # print(train_sampler.num_samples)
        batch_size = batch_size // world_size
        is_shuffle = False

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    drop_last=drop_last,
                                    shuffle=is_shuffle,
                                    pin_memory=True,
                                    sampler=train_sampler,
                                    collate_fn=collate_fn)

    return train_loader, train_sampler


def get_val_loader(args, test_dataset, logger, batch_size, num_workers, distributed, world_size):
    test_sampler = None
    is_shuffle = False
    batch_size = batch_size

    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)
        batch_size = batch_size // world_size

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    drop_last=False,
                                    shuffle=is_shuffle,
                                    pin_memory=True,
                                    sampler=test_sampler)

    return test_loader, test_sampler


def get_loader_LA_seg(args, logger=None):
    if args.test_mode:
        num_workers = args.workers
        valid_dataset = LAseg_Dataset(root=args.LA_Seg_data_base, max_iters=None,
                                      crop_size=(args.roi_x, args.roi_y, args.roi_z), mean=(128, 128, 128), scale=True,
                                      mirror=True, ignore_label=255, ratio_labels=1, split="valid")
        val_loader, val_sampler = get_val_loader(args, valid_dataset, logger, 1, num_workers, False, 1)
        properties = {'name': 'LA_Seg', 'labels': {'0': 'background', '1': 'left_atrium'}}
        return val_loader, properties
    else:
        batch_size = args.batch_size
        num_workers = args.workers
        distributed = args.distributed
        world_size = args.world_size
        train_dataset = LAseg_Dataset(root=args.LA_Seg_data_base, max_iters=None, crop_size=(args.roi_x, args.roi_y, args.roi_z), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, ratio_labels=1, split="train")
        valid_dataset = LAseg_Dataset(root=args.LA_Seg_data_base, max_iters=None, crop_size=(args.roi_x, args.roi_y, args.roi_z), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, ratio_labels=1, split="valid")

        train_loader, train_sampler = get_train_loader(args, train_dataset, logger, batch_size, num_workers, distributed, world_size)
        val_loader, val_sampler = get_val_loader(args, valid_dataset, logger, 1, num_workers, distributed, world_size)
        return [train_loader, val_loader]