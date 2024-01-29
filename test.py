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

from utils import GaussianBlur
from utils import GaussianNoise
from utils import Cutout
from utils import CircularHorizontalShift
from utils import Rotate

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
    parser.add_argument('--shift', default=True)
    parser.add_argument('--is_LSA', default=True)
    parser.add_argument('--set', default='test')
    parser.add_argument('--mode', default='face')
    parser.add_argument('--batch', default=8)
    parser.add_argument('--tta', default=1, type=int)
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

    if args.shift == True:
        channels*=5

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
            base_order  = 1,
            mode        = args.mode, # face, vertex and regular
            samp        = samp,
            channels    = channels,
            dropout     = 0.1,
            emb_dropout = 0.1,
            is_shifted  = args.shift,
            is_LSA      = args.is_LSA
 
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

    tta_transforms = None#transforms.Compose([Rotate(p=0.5),transforms.RandomApply([GaussianBlur(p=1), GaussianNoise(p=1)], p=0.5),])

    P=np.array([])
    #T=np.array([])
    preds_ = None

    for i in range(args.tta):

        if args.mode == 'normal':
            test_data   = dataset[args.dataset](data_dir, args.set, image_size, image_size, None, None, args.shift)
        else:
            test_data   = dataset[args.dataset](data_dir, args.set, image_size[0], image_size[1], None, tta_transforms, args.shift)

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
