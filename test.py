import torch
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from vit_pytorch import ViT
from vit_sphere import ViT_sphere

from torch import nn, optim

import torch
from torchvision import transforms

from datasets.DVSC import DVSC
from datasets.smnist import SMNIST
from datasets.rsmnist import RSMNIST
from datasets.modelnet10 import MODELNET10

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from scipy.spatial.transform import Rotation as R



class GaussianBlur(object):
    def __init__(self, p=1, kernel_size=None, sigma_min=0.1, sigma_max=2.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size   

    def __call__(self, pic):
        if np.random.rand(1) > self.p:
            return pic 
        
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        
        if self.kernel_size == None:
            self.kernel_size = (1+2*np.ceil(2*sigma)).astype('int')  

        pic = cv2.GaussianBlur(np.array(pic), (self.kernel_size, self.kernel_size), sigma)
        return np.array(pic)

class GaussianNoise(object):
    def __init__(self, p=1, mean=None, std=None):
        self.p = p
        self.std = std
        self.mean = mean
        if self.mean == None:
            self.mean = np.random.uniform(0.0, 0.001)
        if self.std == None:
            self.std = np.random.uniform(0.0, 0.03)
        
    def __call__(self, pic):
        if np.random.rand(1) > self.p:
            return pic
        return pic + np.random.randn(*pic.shape) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Cutout(object):
    """
    Applies cutout augmentation
    """

    def __init__(self, p=1, max_size=None, n_squares=None):
        self.p = p
        if max_size == None:
            self.size = np.random.randint(10, 50)
        if n_squares == None:
            self.n_squares = np.random.randint(1, 5)

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be cut
        """
        h, w, _ = pic.shape
        new_image = pic
        if np.random.rand(1) < self.p:
            for _ in range(self.n_squares):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.size // 2, 0, h)
                y2 = np.clip(y + self.size // 2, 0, h)
                x1 = np.clip(x - self.size // 2, 0, w)
                x2 = np.clip(x + self.size // 2, 0, w)
                new_image[y1:y2, x1:x2, :] = 0

        return new_image

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size={0}'.format(self.size)
        format_string += ')'
        return format_string

class CircularHorizontalShift(object):

    """
    Executes a circular horizontal shift 
    """

    def __init__(self, p=0.5, shift_length=None):
        self.p = p
        self.shift_length = shift_length

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be shifted
        """
        if self.shift_length == None:
            self.shift_length = np.random.randint(0, pic.shape[2])

        shifted_img = pic
        if np.random.rand(1) < self.p:
            shifted_img = np.concatenate((pic[:, :, self.shift_length:], pic[:, :, :self.shift_length]), axis=2)
        
        return shifted_img
    
    def __repr__(self):
        format_string = self.__class__.__name__+ '('
        format_string += 'shift_length={0}'.format(self.shift_length)
        format_string += ')'
        return format_string

class Rotate(object):
    def __init__(self, p=0.5, rz=180, rx=15, ry=15):
        self.p = p
        self.rz = rz
        self.rx = rx
        self.ry = ry

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be shifted
        """
        if np.random.rand(1) > self.p:
            return pic

        self.rz = np.random.uniform(-self.rz,self.rz)
        self.rx = np.random.uniform(-self.rx,self.rx)
        self.ry = np.random.uniform(-self.ry,self.ry)
        self.r = R.from_euler("zxy", [self.rz, self.rx, self.ry], degrees=True).as_matrix()

        colors = pic
        dim = pic.shape
        phi, theta = np.meshgrid(np.linspace(0, np.pi, num=dim[0], endpoint=False),
                                 np.linspace(0, 2 * np.pi, num=dim[1], endpoint=False))
        coordSph = np.stack([(np.sin(phi) * np.cos(theta)).T, (np.sin(phi) * np.sin(theta)).T, np.cos(phi).T], axis=2)

        eps = 1e-8
        data = np.array(np.dot(coordSph.reshape((dim[0] * dim[1], 3)), self.r))
        coordSph = data.reshape((dim[0] * dim[1], 3))

        x, y, z = data[:, ].T
        z = np.clip(z, -1 + eps, 1 - eps)

        phi = np.arccos(z)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2 * np.pi
        theta = dim[1] / (2 * np.pi) * theta
        phi = dim[0] / np.pi * phi

        mapped = np.stack([theta.reshape(dim[0], dim[1]), phi.reshape(dim[0], dim[1])], axis=2).astype(np.float32)

        colors = cv2.remap(colors, mapped, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return np.array(colors)


def softmax(x):

    max = np.max(x,axis=1,keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x,axis=1,keepdims=True)
    f_x = e_x / sum
    return f_x
    

def load_model(model, resume):
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    return model

def main():


    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--dataset', default='modelnet10')
    parser.add_argument('--set', default = 'test')
    parser.add_argument('--mode', default='face')
    parser.add_argument('--batch', default=8)
    parser.add_argument('--tta', default=20, type=int)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    os.system('mkdir -p weights')

    dataset = {'smnist': SMNIST, 'dvsc': DVSC, 'rsmnist': RSMNIST, 'modelnet10': MODELNET10}
   
    if args.dataset == 'modelnet10':
        data_dir    = '/home/datasets/ModelNet-10-ERP'
        image_size  = (256, 512)
        patch_size  = 32
        num_classes = 10
        samp        = (8, 16)
        channels    = 12 
    if args.dataset == 'rsmnist':
        data_dir    = '/home/datasets/rsmnist'
        image_size  = (60, 60)
        patch_size  = 10
        num_classes = 10
        samp = (6, 6)
        channels = 3
    if args.dataset == 'smnist':
        data_dir    = '/home/datasets/smnist'
        image_size  = (60, 60)
        patch_size  = 10
        num_classes = 10
        samp = (6, 6)
        channels = 3
    elif args.dataset == 'dvsc':
        image_size  = (384, 384)
        patch_size  = 32
        num_classes = 2
        samp = (12, 12)
        channels = 3

    resume      = args.dataset+'-'+args.mode+'model_last.pth'

    if args.mode == 'normal':
        model = ViT(
            image_size  = image_size,
            patch_size  = patch_size,
            num_classes = num_classes,
            dim         = 512,
            depth       = 4,
            heads       = 8,
            mlp_dim     = 512,
            dropout     = 0.1,
            emb_dropout = 0.1
        )
    else :
        model = ViT_sphere(
            image_size  = image_size,
            patch_size  = patch_size,
            num_classes = num_classes,
            dim         = 512,
            depth       = 4,
            heads       = 8,
            mlp_dim     = 512,
            base_order = 1,
            mode = args.mode, # face, vertex and regular
            samp = samp,
            channels = channels,
            dropout     = 0.1,
            emb_dropout = 0.1
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Trainable parameters", params)

    path    = 'weights/'
    model = load_model(model, os.path.join(path, resume))
    cuda    = args.cuda
    batch   = args.batch


    if cuda:
        model = model.cuda()
    model.eval()

    tta_transforms = transforms.Compose([Rotate(p=0.5),transforms.RandomApply([GaussianBlur(p=1), GaussianNoise(p=1)], p=0.5),])

    P=np.array([])
    #T=np.array([])
    preds_ = None

    for i in range(args.tta):

        if args.mode == 'normal':
            test_data   = dataset[args.dataset](data_dir, args.set, image_size, image_size, None, None)
        else:
            test_data   = dataset[args.dataset](data_dir, args.set, image_size[0], image_size[1], None, tta_transforms)

        test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=False)

        temp_pred = None
        T = np.array([])

        for j, data in enumerate(tqdm(test_loader)):
            img, target = data
            if cuda:
                img    = img.cuda()
                target = target.cuda()
            preds = model(img)

            if temp_pred is None:
                temp_pred = preds.detach().cpu().numpy()
            else:
                temp_pred = np.vstack((temp_pred, preds.detach().cpu().numpy()))


            T = np.concatenate([T,target.cpu().numpy()])

        if preds_ is None:
            preds_ = temp_pred
        else:
            preds_ = preds_ + temp_pred

    print(preds_)
    probabilities = softmax(preds_.T).T
    print(probabilities)
    preds = np.argmax(probabilities, axis =1) 
    print(preds)
    P = np.concatenate([P,preds])

    confusion = confusion_matrix(P, T)

    #df['pred_class'] = P    
    #df.to_csv('dvsc_p_regular.csv')

    print('Confusion Matrix\n')
    print(confusion)
        
    print('\nClassification Report\n')
    print(classification_report(P, T, target_names=test_data.category))


if __name__ == '__main__':
    main()
