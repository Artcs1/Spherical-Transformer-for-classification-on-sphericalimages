
from .base import BaseDataset
import numpy as np
import cv2
import torch
from os import scandir, getcwd
from os.path import abspath,isfile
import os

from torch.utils.data import Dataset

class DVSC(BaseDataset):

    def __init__(self, data_dir, phase, input_h, input_w, down_ratio):
        super(DVSC, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category    = ['cat', 'dog']
        self.num_classes = len(self.category)
        self.cat_ids     = {cat:i for i,cat in enumerate(self.category)}
        self.image_ids   = self.load_img_ids()
        self.len         = len(self.image_ids)
        self.image_path  = os.path.join(data_dir, 'AllImages')

    def load_image(self, index):
        img_id  = self.image_ids[index]
        imgFile = os.path.join(self.image_path, img_id)
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_img_ids(self):
        image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def __len__(self):
        return self.len

    def load_annotation(self, index):
        imgFile    = self.image_ids[index]
        annotation = 0 if "cat" in imgFile else 1
        return annotation

