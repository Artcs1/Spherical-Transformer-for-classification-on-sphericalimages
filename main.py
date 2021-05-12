import timm
import urllib
import numpy as np
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from get_tangentplanes import get_tangent_images
from spherical_distortion.util import *
import matplotlib.pyplot as plt



def coord_3d(X,dim):
    phi   = X[:,1]/dim[1] * np.pi     # phi
    theta = X[:,0]/dim[0] * 2 * np.pi         # theta
    R = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T,np.cos(phi).T], axis=1)

    return R

def multiple_tangentimages(img = 'class_256.jpg', patch = 32, image =384, base_order =1): #TODO: ARRUMAR O ORDEM

    scale_factor = 1
    pi_samp = int(image/patch)

    theta, phi = np.meshgrid(np.linspace(-np.pi,np.pi, num = pi_samp, endpoint=False), np.linspace(-np.pi/2,np.pi/2, num = pi_samp, endpoint=False))
    points = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T,np.cos(phi).T], axis=2)
    points = np.reshape(points, (-1,3))
    points = points.astype('float32')
    points = torch.from_numpy(points)
    icosphere     = generate_icosphere(1)
    #points = icosphere.get_face_barycenters()
    #print(points)
    #spherical_coords = convert_3d_to_spherical(points)
    #print(spherical_coords)
    tex_image = get_tangent_images(img, scale_factor, base_order, points, patch)

    generate_image = np.zeros((image,image,3),dtype='uint8')
    for i in range(tex_image.shape[1]):
        img = tex_image[:, i, ...]
        I2 = torch2numpy(img.byte())
        generate_image[int(patch*(i//pi_samp)):int(patch*(i//pi_samp)+patch),int(patch*(i%pi_samp)):int(patch*(i%pi_samp)+patch),:] = I2

    return generate_image

def test(img_name, model, config, transform, opt, topk = 5):

    if opt == 'tangent':
        img = multiple_tangentimages(img_name)
        img = Image.fromarray(img)
    else:
        img = Image.open(img_name).convert('RGB')
    tensor = transform(img).unsqueeze(0) # transform and add batch dimension
    with torch.no_grad():
        out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, topk)

    return top5_prob, top5_catid


def main():

    opt = 'tangent'

    url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    model = timm.create_model('vit_base_patch32_384', pretrained=True)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # TODO: FOR PARA PERCORRER IMAGENS DE TESTE
    # TODO: SACAR METRICAS COM O GROUND TRUTH

    img_name = 'class_256.jpg'
    topk_prob, topk_catid = test(img_name, model, config, transform, opt, topk=10)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())


if __name__ == '__main__':
    main()
