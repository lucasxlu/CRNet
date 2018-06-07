import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from config.cfg import cfg


class ScutFBPDataset(Dataset):
    """
    SCUT-FBP dataset
    """

    def __init__(self, f_list, f_labels, transform=None):
        self.face_files = f_list
        self.face_score = f_labels.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scut_fbp_dir'], 'SCUT-FBP-%d.jpg' % self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'score': score, 'class': round(score) - 1,
                  'filename': 'SCUT-FBP-%d.jpg' % self.face_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class HotOrNotDataset(Dataset):
    def __init__(self, cv_split=1, train=True, transform=None):
        df = pd.read_csv(
            os.path.join(os.path.split(os.path.abspath(cfg['hotornot_dir']))[0], 'eccv2010_split%d.csv' % cv_split),
            header=None)

        filenames = [os.path.join(cfg['hotornot_dir'], _.replace('.bmp', '.jpg')) for
                     _ in df.iloc[:, 0].tolist()]
        scores = df.iloc[:, 1].tolist()
        flags = df.iloc[:, 2].tolist()

        train_set = OrderedDict()
        test_set = OrderedDict()

        for i in range(len(flags)):
            if flags[i] == 'train':
                train_set[filenames[i]] = scores[i]
            elif flags[i] == 'test':
                test_set[filenames[i]] = scores[i]

        if train:
            self.face_files = list(train_set.keys())
            self.face_scores = list(train_set.values())
        else:
            self.face_files = list(test_set.keys())
            self.face_scores = list(test_set.values())

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        score = self.face_scores[idx]

        if score < -1:
            cls = 0
        elif -1 <= score < 1:
            cls = 1
        elif score >= 1:
            cls = 2

        sample = {'image': image, 'score': score, 'class': cls, 'filename': os.path.basename(self.face_files[idx])}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class JaffeDataset(Dataset):
    """
    Jaffe dataset
    """

    def __init__(self, lb_path='./cvsplit/jaffe.csv', transform=None):
        df = pd.read_csv(lb_path, sep=' ')

        self.files = [os.path.join(cfg['jaffe_dir'], df['PIC'][i].replace('-', '.') + '.' + str(df['#'][i]) + '.tiff')
                      for i
                      in range(len(df['#'].tolist()))]
        self.labels = []
        for index, row in df.iterrows():
            if index > 0:
                self.labels.append(np.argmax(np.array(row[1: 7].tolist())))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = io.imread(self.files[idx])
        label = self.labels[idx]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
