from dataclasses import dataclass


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "image_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    spatial_path: str = "/mnt/data/ssl/data/pretrain/pretrain_patch_list/train_patch_spatial.txt"
    contrast_path: str = "/mnt/data/ssl/data/pretrain/pretrain_patch_list/pair4/train_patch_contrast.txt"
    semantic_path: str = "/mnt/data/ssl/data/pretrain/pretrain_patch_list/pair4/train_patch_semantic.txt"
    img_size = [128, 128, 128]
    patch_size = [16, 16, 16]
    attention_type = 'prefix'
    add_series_data = True
    add_spatial_data = True
    is_subset = False
    series_length = 4
