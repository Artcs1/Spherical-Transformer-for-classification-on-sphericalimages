import torch
import numpy as np
import os

from tqdm import tqdm
from vit_pytorch import ViT
from vit_sphere import ViT_sphere

from torch import nn, optim


from datasets.DVSC import DVSC

from torch.utils.data import DataLoader

def main():

    os.system('mkdir -p weights')

    model = ViT_sphere(
        image_size = 224,
        patch_size = 32,
        num_classes = 2,
        dim = 512,
        depth = 4,
        heads = 8,
        mlp_dim = 512,
        base_order = 1,
        mode = 'vertex', # face, vertex and regular
        dropout = 0.1,
        emb_dropout = 0.1
    )

    cuda    = True
    epochs  = 100
    batch   = 40
    path    = 'weights/model1.pth'

    train_data = DVSC('data/dvsc-data', 'train', 224, 224, None)
    valid_data = DVSC('data/dvsc-data', 'valid', 224, 224, None)

    train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True)

    if cuda:
        model = model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)
    cla_loss  = torch.nn.CrossEntropyLoss()

    print("Training Start")
    for i in range(epochs):
        print("Epoch", i+1)
        L=[]
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, target = data
            if cuda:
                img    = img.cuda()
                target = target.cuda()
            preds = model(img)
            output = cla_loss(preds, target)
            L.append(output.item())
            output.backward()
            optimizer.step()
            optimizer.zero_grad()
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
            probabilities = torch.nn.functional.softmax(preds, dim=0)
            preds = torch.argmax(probabilities, dim =1)
            acc   = torch.sum(torch.where(preds == target, torch.tensor(1, device = preds.device), torch.tensor(0, device = preds.device)))
            sum_acc +=acc

        print("val acc:", sum_acc.item()/total * 100)

    torch.save({'epoch': epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, path)
if __name__ == '__main__':
    main()
