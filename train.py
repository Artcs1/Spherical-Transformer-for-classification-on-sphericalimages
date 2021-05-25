import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from vit_pytorch import ViT
from vit_sphere import ViT_sphere

from torch import nn, optim


from datasets.DVSC import DVSC
from datasets.smnist import SMNIST

from torch.utils.data import DataLoader

def main():


    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--data_dir', default='data/sph_dogs_vs_cats')
    parser.add_argument('--dataset', default='dvsc')
    parser.add_argument('--exp_id', default='sdvsc-adam')
    parser.add_argument('--mode', default='normal')
    parser.add_argument('--batch', default=128)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--optim', default='SGD')
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


    cuda    = args.cuda
    epochs  = args.epochs
    batch   = args.batch
    path    = 'weights/'

    train_data = dataset[args.dataset](args.data_dir, 'train', image_size, image_size, None)
    valid_data = dataset[args.dataset](args.data_dir, 'valid', image_size, image_size, None)

    train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True)

    if cuda:
        model = model.cuda()
    model.train()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)#, momentum=0.9)
    
    cla_loss  = torch.nn.CrossEntropyLoss()


    valid_loss = 1000
    valid_acc  = 0

    print("Training Start")
    T_L = []
    V_L = []
    V_a = []
    for i in range(epochs):
        print("Epoch", i+1)
        model.train()
        L = []
        for i, data in enumerate(tqdm(train_loader)):
            img, target = data
            if cuda:
                img    = img.cuda()
                target = target.cuda()
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
        model.eval()
        for i, data in enumerate(tqdm(valid_loader)):
            img, target = data
            if cuda:
                img    = img.cuda()
                target = target.cuda()
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
            torch.save({'epoch': epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path+args.exp_id+'model_acc.pth')

        if v_l < valid_loss:
            valid_loss = v_l
            torch.save({'epoch': epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path+args.exp_id+'model_loss.pth')

        V_L.append(v_l)
        V_a.append(v_a)
        print("val loss:", v_l)
        print("val acc:", v_a)

    print(T_L)
    plt.plot(T_L, label = 'Total_loss', color = 'blue')
    plt.plot(V_L, label = 'Valid_loss', color = 'red')
    plt.legend(loc="upper left")
    plt.xlabel("num of epochs")
    plt.ylabel("loss")
    plt.savefig(path+args.exp_id+'Learning_Curves.png')
    plt.clf()
    plt.plot(V_a, label = 'Valid_acc', color = 'cyan')
    plt.legend(loc="upper left")
    plt.xlabel("num of epochs")
    plt.ylabel("accuracy")
    plt.savefig(path+args.exp_id+'Val_acc.png')
    

    
    torch.save({'epoch': epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path+args.exp_id+'model_last.pth')
if __name__ == '__main__':
    main()

