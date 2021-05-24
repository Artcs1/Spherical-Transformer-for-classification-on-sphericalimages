import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch import nn, optim

import torch
import timm

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from datasets.DVSC import DVSC
from datasets.smnist import SMNIST
from datasets.simagenet import SIMAGENET

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd

def load_model(model, resume):
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    return model

def main():

    with open("/home/paulo/projects/pytorch-image-models/imagenet_classes.txt","r") as f:
        categories = [s.strip() for s in f.readlines()]
    categories = np.array(categories, dtype=object)

    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--data_dir', default='data/s-mnist')
    parser.add_argument('--dataset', default='smnist')
    parser.add_argument('--resume', default='s2r-mnist-sgd-normalmodel_last.pth')
    parser.add_argument('--mode', default='normal')
    parser.add_argument('--image_size', default=60)
    parser.add_argument('--patch_size', default=10)
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--batch', default=512)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--output_csv')
    args = parser.parse_args()

    model = timm.create_model('vit_base_patch32_384', pretrained=True)

    cuda    = args.cuda
    batch   = args.batch
    dataset = {'smnist': SMNIST, 'dvsc': DVSC, 'simagenet': SIMAGENET}

    model.eval()
    if cuda:
        model = model.cuda()

    P=np.array([],dtype=int)
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    test_data   = dataset[args.dataset](args.data_dir, 'test',transform, 384, 384, None)
    test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=False)

    print("number of files: {}".format(len(test_data.image_ids)))

    print("Test Start")
    for i, data in enumerate(tqdm(test_loader)):
        img, target = data
        if cuda:
            img    = img.cuda()

        with torch.no_grad():
            preds = model(img)
        probabilities = torch.nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(probabilities, dim =1) 
        P = np.concatenate([P,preds.cpu().numpy()])



    pred_class = categories[P]

    result = pd.DataFrame({'pred_class': pred_class})

    result.to_csv(args.output_csv)
   
     
    

if __name__ == '__main__':
    main()
