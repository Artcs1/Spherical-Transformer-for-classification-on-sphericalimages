import torch
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

from datasets.DVSC import DVSC
from datasets.smnist import SMNIST

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def load_model(model, resume):
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    return model

def main():


    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--data_dir', default='data/sph_dogs_vs_cats')
    parser.add_argument('--dataset', default='dvsc')
    parser.add_argument('--resume', default='dvsc-sgd-regularmodel_last.pth')
    parser.add_argument('--set', default = 'test')
    parser.add_argument('--mode', default='regular')
    parser.add_argument('--batch', default=8)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()


    os.system('mkdir -p weights')

    dataset = {'smnist': SMNIST, 'dvsc': DVSC}
    
    if args.dataset == 'smnist':
        image_size  = 60
        patch_size  = 10
        num_classes = 10
        samp = 6
    elif args.dataset == 'dvsc':
        image_size  = 384
        patch_size  = 32
        num_classes = 2
        samp = 12


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
            dropout     = 0.1,
            emb_dropout = 0.1
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Trainable parameters", params)

    path    = 'weights/'
    model = load_model(model, os.path.join(path, args.resume))
    cuda    = args.cuda
    batch   = args.batch

    test_data   = dataset[args.dataset](args.data_dir, args.set, image_size, image_size, None)
    test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=False)

    if cuda:
        model = model.cuda()
    model.eval()

    P=np.array([])
    T=np.array([])
    
    #df = pd.read_csv("dvsc.csv")

    for i, data in enumerate(tqdm(test_loader)):
        img, target = data
        if cuda:
            img    = img.cuda()
            target = target.cuda()
        preds = model(img)
        probabilities = torch.nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(probabilities, dim =1) 
        P = np.concatenate([P,preds.cpu().numpy()])
        T = np.concatenate([T,target.cpu().numpy()])

    confusion = confusion_matrix(P, T)

    #df['pred_class'] = P    
    #df.to_csv('dvsc_p_regular.csv')

    print('Confusion Matrix\n')
    print(confusion)
        
    print('\nClassification Report\n')
    print(classification_report(P, T, target_names=test_data.category))


if __name__ == '__main__':
    main()
