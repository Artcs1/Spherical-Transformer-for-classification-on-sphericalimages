import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

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
    parser.add_argument('--data_dir', default='data/s-mnist')
    parser.add_argument('--dataset', default='smnist')
    parser.add_argument('--resume', default='s2r-mnist-sgd-normalmodel_last.pth')
    parser.add_argument('--mode', default='normal')
    parser.add_argument('--image_size', default=60)
    parser.add_argument('--patch_size', default=10)
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--batch', default=64)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()


    os.system('mkdir -p weights')

    if args.mode == 'normal':
        model = ViT(
            image_size  = args.image_size,
            patch_size  = args.patch_size,
            num_classes = args.num_classes,
            dim         = 512,
            depth       = 4,
            heads       = 8,
            mlp_dim     = 512,
            dropout     = 0.1,
            emb_dropout = 0.1
        )
    else :
        model = ViT_sphere(
            image_size  = args.image_size,
            patch_size  = args.patch_size,
            num_classes = args.num_classes,
            dim         = 512,
            depth       = 4,
            heads       = 8,
            mlp_dim     = 512,
            base_order = 1,
            mode = args.mode, # face, vertex and regular
            samp = 6,
            dropout     = 0.1,
            emb_dropout = 0.1
        )

    path    = 'weights/'
    model = load_model(model, os.path.join(path, args.resume))
    cuda    = args.cuda
    batch   = args.batch
    dataset = {'smnist': SMNIST, 'dvsc': DVSC}

    test_data   = dataset[args.dataset](args.data_dir, 'test', 60, 60, None)
    test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)

    if cuda:
        model = model.cuda()
    model.eval()

    P=np.array([])
    T=np.array([])
    

    sum_acc = 0
    print("Test Start")
    for i, data in enumerate(tqdm(test_loader)):
        img, target = data
        if cuda:
            img    = img.cuda()
            target = target.cuda()
        preds = model(img)
        probabilities = torch.nn.functional.softmax(preds, dim=0)
        preds = torch.argmax(probabilities, dim =1) 
        P = np.concatenate([P,preds.cpu().numpy()])
        T = np.concatenate([T,target.cpu().numpy()])
        acc   = torch.sum(torch.where(preds == target, torch.tensor(1, device = preds.device), torch.tensor(0, device = preds.device)))
        sum_acc +=acc

    confusion = confusion_matrix(P, T)
    print('Confusion Matrix\n')
    print(confusion)
        
    print('\nClassification Report\n')
    print(classification_report(P, T, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))


if __name__ == '__main__':
    main()
