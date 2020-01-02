# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import pickle
import pandas as pd

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root_path='./MiniImagenet/', transform=None, is_train=False, is_test=False, target_transform=None, select_class=None, k_shot=None, seed=0):
        self.transform = transform
        self.target_transform = target_transform
        np.random.seed(seed)

        if is_train:
            train_folder = root_path + 'miniImageNet_category_split_train_val.pickle'
            print('load train dataset')
        elif is_test:
            train_folder = root_path + 'miniImageNet_category_split_test.pickle'
            print('load test dataset')
        else:
            train_folder = root_path + 'miniImageNet_category_split_validation.pickle'
            print('load validation dataset')

        data = load_data(train_folder)

        self.train_roots = data['data']
        self.train_labels = data['labels']
        print(len(self.train_roots))


    def __getitem__(self, index):
        image = self.train_roots[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        label = self.train_labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.train_roots)

class GeneratorSampler(Sampler):
    def __init__(self, num_of_class, num_per_class, n_class):
        self.num_per_class = num_per_class
        self.num_of_class = num_of_class
        self.n_class = n_class

    def __iter__(self):
        feature_list = range(600)
        class_list = range(self.n_class)
        class_list = np.random.choice(class_list, self.num_of_class, replace=False)
        feature_idx = np.random.choice(feature_list, self.num_per_class, replace=False)
        batch = []
        for j in class_list:
            feature_idx = np.random.choice(feature_list, self.num_per_class, replace=False)
            batch.append([i+j*600 for i in feature_idx])
        batch = [item for sublist in batch for item in sublist]

        return iter(batch)

    def __len__():
        return 1

class GeneratorSupportSampler(Sampler):
    def __init__(self, n_class, n_support_pairs):
        self.n_class = n_class
        self.n_support_pairs = n_support_pairs

    def __iter__(self):
        feature_list = range(600)
        class_list = range(self.n_class)
        class_list = np.random.choice(class_list, self.n_support_pairs)
        feature_idx = np.random.choice(feature_list, 2, replace=False)
        batch = []
        for j in class_list:
            feature_idx = np.random.choice(feature_list, 2, replace=False)
            batch.append([i+j*600 for i in feature_idx])

        batch = [item for sublist in batch for item in sublist]

        return iter(batch)

    def __len__():
        return 1

def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data
