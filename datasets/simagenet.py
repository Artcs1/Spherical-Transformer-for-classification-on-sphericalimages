
from .base_imagenet import BaseDataset
import numpy as np
import cv2
import torch
from os import scandir, getcwd
from os.path import abspath,isfile
import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class SIMAGENET(BaseDataset):

    def __init__(self, data_dir, phase, transform, input_h, input_w, down_ratio):
        super(SIMAGENET, self).__init__(data_dir, phase, transform, input_h, input_w, down_ratio)
        with open("imagenet_classes.txt", "r") as f:
            self.category = [s.strip() for s in f.readlines()]
        self.num_classes = len(self.category)
        self.cat_ids     = {cat:i for i,cat in enumerate(self.category)}
        self.image_ids   = self.load_img_ids()
        self.len         = len(self.image_ids)
        self.image_path  = data_dir

    def load_image(self, index):
        img_id  = self.image_ids[index]
        print(img_id)
        imgFile = os.path.join(self.image_path, img_id)
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = Image.open(imgFile).convert('RGB')
        return img

    def load_img_ids(self):
     
        image_lists = list(pd.read_csv('~/projects/sphere_projection/sph_imagenet_5k.csv')['file'])
        return image_lists

    def __len__(self):
        return self.len

    def load_annotation(self, index):
        imgFile    = self.image_ids[index]
        annotation = imgFile.split('/')[0]
        return annotation

