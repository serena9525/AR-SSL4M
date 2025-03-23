import math
import os
import pdb
import torch
import numpy as np

from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai import data, transforms
from monai.data import load_decathlon_datalist, load_decathlon_properties
from monai.config import DtypeLike, KeysCollection


class ScaleIntensityRanged_select(transforms.ScaleIntensityRanged):
    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, a_min, a_max, b_min, b_max, clip, dtype, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def __call__(self, data):
        d = dict(data)
        name = data['name']
        key = name.split('/')[-3]

        if key in ['Task03_Liver']:
            self.scaler = transforms.ScaleIntensityRange(-21, 189, self.b_min, self.b_max, self.clip)
        elif key in ['Task06_Lung']:
            self.scaler = transforms.ScaleIntensityRange(-1000, 1000, self.b_min, self.b_max, self.clip)
        elif key in ['Task07_Pancreas']:
            self.scaler = transforms.ScaleIntensityRange(-87, 199, self.b_min, self.b_max, self.clip)
        elif key in ['Task08_HepaticVessel']:
            self.scaler = transforms.ScaleIntensityRange(0, 230, self.b_min, self.b_max, self.clip)
        elif key in ['Task09_Spleen']:
            self.scaler = transforms.ScaleIntensityRange(-57, 175, self.b_min, self.b_max, self.clip)
        elif key in ['Task10_Colon']:
            self.scaler = transforms.ScaleIntensityRange(-57, 175, self.b_min, self.b_max, self.clip)
        else:
            self.scaler = transforms.ScaleIntensityRange(self.a_min, self.a_max, self.b_min, self.b_max, self.clip)
        d = super().__call__(d)
        return d


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args, logger=None):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    sample_ratios = args.sample_ratios
    if sample_ratios is not None:
        sample_ratios = [int(x) for x in sample_ratios.split(',')]

    property_keys = ['name', 'modality', 'labels', 'numTraining', 'numValidation']
    properties = load_decathlon_properties(datalist_json, property_keys) # Task01_BrainTumour
    is_multi_class = len(properties['labels'])>2
    properties['labels'] = {str(int(k)):v for k,v in properties['labels'].items()}

    train_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"], reader='ITKReader'),
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),

            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged_select(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),

            transforms.RandCropByLabelClassesd(
                keys=["image", "label", ],
                label_key="label",
                spatial_size=roi_size,
                ratios=sample_ratios,
                num_classes=len(properties['labels']),
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ) if is_multi_class else
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1, neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"], reader='ITKReader'),
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged_select(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"], reader='ITKReader'),
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            # transforms.Spacingd(
            #     keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z),
            #     mode=("bilinear")
            # ),
            ScaleIntensityRanged_select(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
        for item in test_files:
            item.update({'name':item['image']})
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=False,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

        for item in datalist:
            item.update({'name':item['image']})
        data_ratio = args.ratio
        datalist = datalist[:int(len(datalist)*data_ratio)]
        if args.rank == 0:
            logger.info("number of train subjects: {}, ratio: {}".format(len(datalist), data_ratio))
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=args.cache_num, cache_rate=args.cache_rate, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory= True,
        )

        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        for item in val_files:
            item.update({'name':item['image']})
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
        )
        loader = [train_loader, val_loader]

    return loader, properties
