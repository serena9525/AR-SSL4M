import os
import csv
import pdb
import random
random.seed(0)

from util import load_string_list, save_string_list


def read_txt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = [f.strip() for f in data]
    return data


def readCSV(filename):
    lines = []

    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def gen_spatial_list(data_gen_path, save_path):
    name_lst = ['patch_random_spatial']

    data_list = []
    for name in name_lst:
        data_list += [f'{data_gen_path}/{name}/{x}' for x in os.listdir(f'{data_gen_path}/{name}')]

    save_string_list(f'{save_path}/train_patch_spatial.txt', data_list)


def gen_contrast_list(data_gen_path, save_path):
    name_lst = ['patch_random_contrast']

    data_list = []
    for name in name_lst:
        data_list += [f'{data_gen_path}/{name}/{x}' for x in os.listdir(f'{data_gen_path}/{name}') if x.endswith('t1n.npy')]

    # pair4
    data_list_pair4 = [f'{x},{x.replace("t1n.npy", "t1c.npy")},{x.replace("t1n.npy", "t2w.npy")},{x.replace("t1n.npy", "t2f.npy")}' for x in data_list]
    save_string_list(f'{save_path}/pair4/train_patch_contrast.txt', data_list_pair4)


def gen_semantic_list(data_gen_path, save_path):
    name_lst = ['patch_random_semantic']

    all_data_list = []
    for name in name_lst:
        for num in range(8):
            data_list = [f'{data_gen_path}/{name}/{x}' for x in os.listdir(f'{data_gen_path}/{name}') if x.endswith(f'_{num + 1}.npy')]

            for series_num in range(20000):
                choose_list = random.sample(data_list, 4)
                choose_str = ','.join(choose_list)
                all_data_list.append(choose_str)

    save_string_list(f'{save_path}/pair4/train_patch_semantic.txt', all_data_list)


def main():
    """ base_path: your base path"""
    base_path = '/mnt/data/ssl/data/pretrain/'

    save_path = f'{base_path}/pretrain_patch_list/'

    gen_spatial_list(base_path, save_path)
    gen_contrast_list(base_path, save_path)
    gen_semantic_list(base_path, save_path)


if __name__ == "__main__":
    main()
