import torch
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from vit_pytorch import ViT
from vit_sphere import ViT_sphere

from torch import nn, optim
from torchvision import transforms


from datasets.DVSC import DVSC
from datasets.smnist import SMNIST
from datasets.rsmnist import RSMNIST
from datasets.modelnet10 import MODELNET10

from torch.utils.data import DataLoader



from scipy.spatial.transform import Rotation as R


from utils import GaussianBlur
from utils import GaussianNoise
from utils import Cutout
from utils import CircularHorizontalShift
from utils import Rotate

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif val_acc - self.best_acc > self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        elif val_acc - self.best_acc <= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class CustomScheduler(object):

    def __init__(self, optimizer, change_epoch=30, initial_lr=1e-3, factor=0.7, min_lr=1e-7, verbose=False):
        super(CustomScheduler, self).__init__()
        self.change_epoch = change_epoch
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.factor = factor
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.verbose = verbose

    def step(self, epoch):
        if epoch < self.change_epoch: # warmup
            self.current_lr = self.initial_lr * np.exp(2*(epoch/self.change_epoch -1))
        elif epoch < 2*self.change_epoch: # second stage largest lr
            self.current_lr =  self.initial_lr
        else: # last stage exponential
            self.current_lr = self.current_lr * np.power(self.factor, epoch/(2*self.change_epoch))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
            if self.verbose:
                print(f'Epoch {epoch+1}:  learning rate to {self.current_lr}')


def main():


    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--dataset', default='modelnet10')
    parser.add_argument('--shift', default=True)
    parser.add_argument('--is_LSA', default=True)
    parser.add_argument('--mode', default='face')
    parser.add_argument('--batch', default=16)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--optim', default='Adam')
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
        samp        = (8,16)
        channels    = 12
    if args.dataset == 'smnist':
        data_dir    = '/home/datasets/smnist'
        image_size  = (60, 60)
        patch_size  = 10
        num_classes = 10
        samp = (6, 6)
        channels = 3
    elif args.dataset == 'rsmnist':
        data_dir    = '/home/datasets/rsmnist'
        image_size  = (60, 60)
        patch_size  = 10
        num_classes = 10
        samp = (6, 6)
        channels = 3
    elif args.dataset == 'dvsc':
        data_dir    = 'data/dvsc'
        image_size  = (384, 384)
        patch_size  = 32
        num_classes = 2
        samp = (12, 12)
        channels = 3

    if args.shift == True:
        channels*=5

    exp_id = args.dataset + '-' + args.mode

    if args.mode == 'normal':
        model = ViT(
            image_size  = (image_size[0]+ image_size[1])//2,
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


    cuda    = args.cuda
    epochs  = args.epochs
    batch   = args.batch
    path    = 'weights/'





    train_transforms = transforms.Compose([Rotate(p=0.5),transforms.RandomApply([GaussianBlur(p=0.5),GaussianNoise(p=0.5)], p=0.5),Cutout(p=0.5)]) # blur e noise estavam com p=1

    if args.mode == 'normal':
        #train_data = dataset[args.dataset](data_dir, 'train', (image_size[0]+ image_size[1])//2, (image_size[0]+ image_size[1])//2, None, train_transforms)
        train_data = dataset[args.dataset](data_dir, 'train', (image_size[0]+ image_size[1])//2, (image_size[0]+ image_size[1])//2, None, None, args.shift)
        valid_data = dataset[args.dataset](data_dir, 'valid', (image_size[0]+ image_size[1])//2, (image_size[0]+ image_size[1])//2, None, None, args.shift)
    else:
        #train_data = dataset[args.dataset](data_dir, 'train', image_size[0], image_size[1], None, train_transforms)
        train_data = dataset[args.dataset](data_dir, 'train', image_size[0], image_size[1], None, None, args.shift)
        valid_data = dataset[args.dataset](data_dir, 'valid', image_size[0], image_size[1], None, None, args.shift)

    train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True)

    if cuda:
        model = model.cuda()
    model.train()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.999))
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(0.9, 0.999),weight_decay=1e-4)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9,weight_decay=1e-4)
    else:
        raise Exception('Optimizer not supported.')

    #if args.optim == 'SGD':
    #    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #else:
    #    optimizer = optim.Adam(model.parameters(), lr=1e-3)#, momentum=0.9)

    lr_scheduler = CustomScheduler(optimizer, change_epoch=25, initial_lr=1e-4, min_lr=1e-7,factor=0.9,verbose=True)
    es = EarlyStopping(patience=25)
    
    cla_loss  = torch.nn.CrossEntropyLoss()


    valid_loss = 1000
    valid_acc  = 0

    print("Training Start")
    T_L = []
    V_L = []
    V_a = []
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        model.train()
        L = []
        for i, data in enumerate(tqdm(train_loader)):
            img, target = data
            #print(img.shape)
            if cuda:
                img    = img.cuda()
                target = target.long().cuda()
            preds = model(img)
            output = cla_loss(preds, target)
            L.append(output.cpu().item())
            output.backward()
            optimizer.step()
            optimizer.zero_grad()

        T_L.append(np.mean(L))
        print("train loss:", np.mean(L))

        sum_acc = 0
        total   = len(valid_data)
        if (epoch+1) % 1 == 0:
            model.eval()
            for i, data in enumerate(tqdm(valid_loader)):
                img, target = data
                if cuda:
                    img    = img.cuda()
                    target = target.long().cuda()
                preds = model(img)
                L.append(cla_loss(preds,target).item())
                probabilities = torch.nn.functional.softmax(preds, dim=1)
                preds = torch.argmax(probabilities, dim =1)
                acc   = torch.sum(torch.where(preds == target, torch.tensor(1, device = preds.device), torch.tensor(0, device = preds.device)))
                sum_acc +=acc
            
            v_l = np.mean(L) 
            v_a = sum_acc.item()/total*100
            
            if v_a > valid_acc:
                valid_acc = v_a
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path+exp_id+'model_acc.pth')
    
            if v_l < valid_loss:
                valid_loss = v_l
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path+exp_id+'model_loss.pth')
    
            V_L.append(v_l)
            V_a.append(v_a)
            print("val loss:", v_l)
            print("val acc:", v_a)

        lr_scheduler.step(epoch)

        es(torch.as_tensor(valid_acc))
        if es.early_stop:
            break
    
    print(T_L)
    plt.plot(T_L, label = 'Total_loss', color = 'blue')
    plt.plot(V_L, label = 'Valid_loss', color = 'red')
    plt.legend(loc="upper left")
    plt.xlabel("num of epochs")
    plt.ylabel("loss")
    plt.savefig(path+exp_id+'Learning_Curves.png')
    plt.clf()
    plt.plot(V_a, label = 'Valid_acc', color = 'cyan')
    plt.legend(loc="upper left")
    plt.xlabel("num of epochs")
    plt.ylabel("accuracy")
    plt.savefig(path+exp_id+'Val_acc.png')
    

    
    torch.save({'epoch': epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path+exp_id+'model_last.pth')
if __name__ == '__main__':
    main()

