from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision as tv
import pandas as pd
import os
# import cv2
from matplotlib import image
# import torch
import numpy as np
# import random
# import matplotlib.pyplot as plt


class Dataset_classification(Dataset):
    def __init__(self,
                 pos_dir,
                 neg_dir,
                 mode,
                 transform=None,
                 seed=2023):
        self.transform = transform
        # pos_names = [os.path.join(pos_dir, val) for val in os.listdir(pos_dir) if '.png' in val]
        # neg_names = [os.path.join(neg_dir, val) for val in os.listdir(neg_dir) if '.png' in val]
        pos_names = [os.path.join(pos_dir, val) for val in os.listdir(pos_dir)]
        neg_names = [os.path.join(neg_dir, val) for val in os.listdir(neg_dir)][0:len(pos_names)]
        assert len(pos_names) == len(neg_names)
        n_files_single = len(pos_names)
        np.random.seed(seed)
        np.random.shuffle(pos_names)
        np.random.shuffle(neg_names)
        # print("pos")
        # for i in range(10,13):
        #     print(pos_names[i])
        # print("neg")
        # for i in range(10,13):
        #     print(neg_names[i])
        # data splits
        if mode == "train":
            pos_names = pos_names[0:int(n_files_single * 0.7)]
            neg_names = neg_names[0:int(n_files_single * 0.7)]
        elif mode == "valid":
            pos_names = pos_names[int(n_files_single * 0.7):int(n_files_single * 0.85)]
            neg_names = neg_names[int(n_files_single * 0.7):int(n_files_single * 0.85)]
        else:
            pos_names = pos_names[int(n_files_single * 0.85):]
            neg_names = neg_names[int(n_files_single * 0.85):]
        self.filenames = pos_names + neg_names
        labels_p = np.ones(len(pos_names), dtype=int)
        labels_n = np.zeros(len(neg_names), dtype=int)
        self.labels = np.concatenate((labels_p, labels_n), axis=0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img = image.imread(img_name)
        # img = np.mean(img, axis=2)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # tailor
        img = img[184:808, 40:1185]
        # stack to three channels
        img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
        # plt.imshow(img, cmap='gray', vmin=0, vmax=255)

        # down sample; normalization
        if self.transform:
            img = self.transform(img)

        return {'images': img,
                'labels': self.labels[idx]
                }


class Dataset_regression(Dataset):  # TODO
    def __init__(self,
                 pos_dir,
                 label_excel,
                 mode,
                 seed=2023,
                 transform=None):
        self.transform = transform
        # pos_names = [os.path.join(pos_dir, val) for val in os.listdir(pos_dir) if '.png' in val]
        # neg_names = [os.path.join(neg_dir, val) for val in os.listdir(neg_dir) if '.png' in val]
        pos_names = [os.path.join(pos_dir, val) for val in os.listdir(pos_dir)]
        n_files_single = len(pos_names)
        np.random.seed(seed)
        np.random.shuffle(pos_names)
        # data splits
        if mode == "train":
            pos_names = pos_names[0:int(n_files_single * 0.7)]
        elif mode == "valid":
            pos_names = pos_names[int(n_files_single * 0.7):int(n_files_single * 0.85)]
        else:
            pos_names = pos_names[int(n_files_single * 0.85):]
        self.filenames = pos_names
        df = pd.read_csv(label_excel)  # TODO: RHC-MPAP
        drop_name = []
        for col in df.columns:
            if "Name" not in col and 'MPAP' not in col:
                drop_name.append(col)
            if 'MPAP1' in col:
                drop_name.append(col)
        df.drop(columns=drop_name, axis=1, inplace=True)
        self.labels = df
        for col in df.columns:
            if 'MPAP' in col:
                self.label_key = col
            if 'code' in col:
                self.index_column = col
        print(self.labels.dtypes)
        print("name code:", self.index_column)        
        print("name code:", self.label_key)        
        self.labels[self.label_key] = self.labels[self.label_key].astype(float)
        self.labels[self.index_column] = self.labels[self.index_column].astype(float)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img = image.imread(img_name)

        if self.transform:
            img = self.transform(img)

        # label
        img_name_num = float(os.path.split(img_name)[1].split('.')[0])
        idx = self.labels[self.labels[self.index_column] == img_name_num].index
        assert len(idx) == 1
        val = self.labels[self.label_key].iloc[idx[0]]

        return {'images': img,
                'values': val  # 0 or 1
                }


if __name__ == '__main__':
    # test classification loader
    # pos_file = '../data/pos'
    # neg_file = '../data/neg'
    # trainTransform = tv.transforms.Compose([tv.transforms.ToTensor(),
    #                                         tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
    #
    # data_loader = Dataset_classification(pos_file, neg_file, mode="train", transform=trainTransform)
    # loader = DataLoader(data_loader, batch_size=1, num_workers=6, pin_memory=True, shuffle=True)
    # # data_iter = iter(loader)
    # for i, data_sample in enumerate(loader):
    #     print(data_sample)
    #
    # regression data loader
    pos_dir = os.path.join('/home/chain/gpu', 'data/processed_pos/fig')
    csv_file = os.path.join('/home/chain/gpu', 'data/processed_pos/label_3.csv')
    trainTransform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                            tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])

    data_loader = Dataset_regression(pos_dir, csv_file, mode="train", seed=2023, transform=trainTransform)
    loader = DataLoader(data_loader, batch_size=2, num_workers=6, pin_memory=False, shuffle=True)
    # data_iter = iter(loader)
    for i, data_sample in enumerate(loader):
        print(data_sample)
        break

