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


    cuda    = args.cuda
    epochs  = args.epochs
    batch   = args.batch
    path    = 'weights/'





    train_transforms = transforms.Compose([Rotate(p=0.5),transforms.RandomApply([GaussianBlur(p=0.5),GaussianNoise(p=0.5)], p=0.5),Cutout(p=0.5)]) # blur e noise estavam com p=1

    if args.mode == 'normal':
        train_data = dataset[args.dataset](data_dir, 'train', (image_size[0]+ image_size[1])//2, (image_size[0]+ image_size[1])//2, None, train_transforms)
        valid_data = dataset[args.dataset](data_dir, 'valid', (image_size[0]+ image_size[1])//2, (image_size[0]+ image_size[1])//2, None, None)
    else:
        train_data = dataset[args.dataset](data_dir, 'train', image_size[0], image_size[1], None, train_transforms)
        valid_data = dataset[args.dataset](data_dir, 'valid', image_size[0], image_size[1], None, None)

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

