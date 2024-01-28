import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from get_tangentplanes import get_tangent_images, get_tangent_images2
from spherical_distortion.util import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

from utils import GaussianBlur
from utils import GaussianNoise
from utils import Cutout
from utils import CircularHorizontalShift
from utils import Rotate

from utils import Deterministic_Rotate


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PatchShifting(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):


        print(x.shape)
        rot_1 = Deterministic_Rotate(0,15,0)
        x1    = torch.from_numpy(rot_1(x.cpu().numpy())).cuda()

        print(x1.shape)

        x_cat = torch.cat([x,x1,x,x,x], dim = 1)
        out = x_cat
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))    
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)


        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., is_LSA = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA = is_LSA)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TangentPlane(nn.Module):
    def __init__(self, points, base_order, patch_size, mode, samp, scale_factor=1):
        super().__init__()
        self.points       = points
        self.base_order   = base_order
        self.patch        = patch_size
        self.scale_factor = scale_factor
        self.mode         = mode
        self.samp         = samp

        if mode == 'face':
            num_patches = 20*(4**base_order)
        elif mode =='vertex':
            num_patches = 12
            for i in range(base_order):
                num_patches += 30*(4**(base_order-1))
        elif mode == 'regular':
            num_patches = self.samp[0]*self.samp[1]

        self.num_patches  = num_patches


    def forward(self,x):
        #print('x.shape')
        #print(x.shape)
        for i in range(x.shape[0]):
            img = x[i,:,:,:]
            tex_image = get_tangent_images2(img, self.base_order, self.points, self.patch).float()
            tex_image = torch.transpose(tex_image,0,1)

            tex_image = tex_image.reshape(-1,self.patch*self.patch*x.shape[1])
            if i == 0:
                O = tex_image
            else:
                O = torch.cat((O,tex_image),0)
        O = O.view(-1, self.num_patches,self.patch*self.patch*x.shape[1])
        #print(O.shape)
        O = O.float()
        return O


class ViT_sphere(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, base_order, mode = 'face', samp = (12, 12), pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., is_shifted = False, is_LSA=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
 
        if mode == 'face':
            S = generate_icosphere(base_order).get_face_barycenters()
            S = convert_3d_to_spherical(S)
            ind = np.lexsort((S[:,0],S[:,1]))
            S   = S[ind]
        elif mode == 'vertex':
            S = generate_icosphere(base_order).get_vertices()
            S = convert_3d_to_spherical(S)
            ind = np.lexsort((S[:,0],S[:,1]))
            S   = S[ind]
        elif mode == 'regular':
            theta   = np.linspace(-np.pi/2, np.pi/2, num = samp[0], endpoint = False)
            phi = np.linspace(-np.pi, np.pi, num = samp[1], endpoint = False)
            c, d = np.meshgrid(phi,theta)
            S = np.stack((c.flat,d.flat),axis=1)
            S = torch.from_numpy(S.astype('float32'))

        points = convert_spherical_to_3d(S).cuda()

        if mode == 'face':
            num_patches = 20*(4**base_order)
        elif mode =='vertex':
            num_patches = 12
            for i in range(base_order):
                num_patches += 30*(4**(base_order-1))
        elif mode =='regular':
            num_patches = samp[0]*samp[1]


        # num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'



        self.to_patch_embedding = nn.Sequential(
            TangentPlane(points, base_order, patch_height, mode, samp = samp),
            #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.is_shifted = is_shifted
        self.patch_shifting = PatchShifting()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, is_LSA=is_LSA)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        #if self.is_shifted == True:
        #    img = self.patch_shifting(img)
        #print(img_.shape)
        #print(img.shape)

        x = self.to_patch_embedding(img)
        #print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

