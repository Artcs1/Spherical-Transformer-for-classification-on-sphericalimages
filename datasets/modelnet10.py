
from .base import BaseDataset
import numpy as np
import cv2
import torch
from os import scandir, getcwd
from os.path import abspath,isfile
import os
import pandas as pd

from torch.utils.data import Dataset

class MODELNET10(BaseDataset):

    def __init__(self, data_dir, phase, input_h, input_w, down_ratio, transform):
        super(MODELNET10, self).__init__(data_dir, phase, input_h, input_w, down_ratio, transform)
        self.category    = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table' , 'toilet']
        self.num_classes = len(self.category)
        self.cat_ids     = {cat:i for i,cat in enumerate(self.category)}
        self.image_ids   = self.load_img_ids()
        self.len         = len(self.image_ids)
        self.image_path  = data_dir
        df               = pd.read_csv(self.data_dir+'/'+ self.phase + '_npy/'+self.phase+'_csv.csv')
        test_keys        = df.values[:,0]
        test_values      = df.values[:,1]
        self.dict        = {test_keys[i]: test_values[i] for i in range(len(test_keys))}

    def load_image(self, index):
        img_id  = self.image_ids[index]
        imgFile = os.path.join(self.image_path, self.phase + '_npy', img_id)
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = np.load(imgFile)
        return img

    def load_img_ids(self):
        image_set_index_file = os.path.join(self.data_dir, self.phase + '_npy')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        entries = os.listdir(image_set_index_file)
        image_list = [entry for entry in entries if '.npy' in entry]
        return image_list

    def __len__(self):
        return self.len

    def load_annotation(self, index):
        annotation = self.dict[self.image_ids[index]]
        return annotation


