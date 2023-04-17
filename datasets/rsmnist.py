
from .base import BaseDataset
import numpy as np
import cv2
import torch
from os import scandir, getcwd
from os.path import abspath,isfile
import os

from torch.utils.data import Dataset

class RSMNIST(BaseDataset):

    def __init__(self, data_dir, phase, input_h, input_w, down_ratio):
        super(RSMNIST, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category    = ['0', '1','2','3','4','5','6','7','8','9']
        self.num_classes = len(self.category)
        self.cat_ids     = {cat:i for i,cat in enumerate(self.category)}
        self.I           = np.load(self.data_dir+'/s2_r_mnist', allow_pickle=True)
        self.image_ids   = self.load_img_ids()
        self.len         = len(self.image_ids)
        self.image_path  = data_dir

    def load_image(self, index):
        if self.phase == 'train':
            I = self.I['train']['images']
        elif self.phase == 'valid':
            I = self.I['train']['images']
        elif self.phase == 'test':
            I = self.I['test']['images']

        img = I[int(self.image_ids[index]),...]
        img = np.stack((img,img,img),2)
        return img

    def load_img_ids(self):
        #I  = np.load(self.data_dir+'/s2_mnist', allow_pickle=True)
        #I  = I['train']['images'].shape[0]
        if self.phase == 'train':
            image_lists = [str(i) for i in range(50000)]
        elif self.phase == 'valid':
            image_lists = [str(i) for i in range(50000,60000)]
        elif self.phase == 'test':
            image_lists = [str(i) for i in range(10000)]

        return image_lists

    def __len__(self):
        return self.len

    def load_annotation(self, index):
        if self.phase == 'train':
            I = self.I['train']['labels']
        elif self.phase == 'valid':
            I = self.I['train']['labels']
        elif self.phase == 'test':
            I = self.I['test']['labels']

        annotation = I[int(self.image_ids[index])]
        return annotation


