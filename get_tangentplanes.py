"""
    PROGRAMA PARA CALCULAR KEYPOINTS DE DOS IMAGENES 512x1024, ASI MISMO DE SUS CORRESPONDENCIAS
    POR UN KNN BILATERAL

"""


import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, unresample
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os

import numpy as np
import _spherical_distortion_ext._mesh as _mesh
import argparse

import matplotlib.pyplot as plt

def tangent_images(
    input,  # [B] x C x H x W
    base_order,
    sample_order,
    points,
    num_samples,
    interpolation=InterpolationType.BISPHERICAL,
    return_mask=False):

    assert input.dim() in [3, 4], \
        'input must be a 3D or 4D tensor (input.dim() == {})'.format(input.dim())

    sample_map = tangent_sample_map(
        input.shape[-2:], base_order, sample_order, points, num_samples)

    if input.is_cuda:
        sample_map = sample_map.to(input.get_device())

    # Resample to the tangent images
    tangent_images = unresample(input, sample_map, interpolation)

    # Reshape to a separate each patch
    tangent_images = tangent_images.view(*tangent_images.shape[:-1],
                                         num_samples, num_samples)
    if return_mask:
        return tangent_images, compute_icosahedron_face_mask(
            base_order, sample_order)
    return tangent_images

# -----------------------------------------------------------------------------

def tangent_sample_map(
    image_shape, base_order, sample_order, points, num_samples):

    assert sample_order >= base_order, 'Sample order must be greater than or equal to the base order ({} <{ })'.format(
        sample_order, base_order)

    # Generate the spherical sample map
    spherical_sample_map = get_sample_map(
        base_order, sample_order, points, num_samples)

    # Produces a sample map to turn the image into tangent planes
    image_sample_map = convert_spherical_to_image(spherical_sample_map,
                                                  image_shape)

    # Returns F_base x num_samples^2 x 2 sample map
    return image_sample_map.squeeze(0)


def get_sample_map(base_order, sample_order, points, num_samples):

    assert sample_order >= base_order, 'Sample order must be greater than or equal to the base order ({} <{ })'.format(
        sample_order, base_order)

    # Generate the base icosphere
    base_sphere = generate_icosphere(base_order)

    # Get sampling resolution
    sampling_resolution = get_sampling_resolution(base_order)


    # Generate spherical sample map s.t. each face is projected onto a tangent grid of size (num_samples x num_samples) and the samples are spaced (sampling_resolution/num_samples x sampling_resolution/num_samples apart)
    spherical_sample_map = get_gnomonic_projection(
        base_sphere,
        num_samples,
        num_samples,
        sampling_resolution / num_samples,
        sampling_resolution / num_samples,
        points)

    return spherical_sample_map

def get_gnomonic_projection(icosphere,
                                kh,
                                kw,
                                res_lat,
                                res_lon,
                                points,
                                source='vertex'):
    '''
    Returns a map of gnomonic filters with shape (kh, kw) and spatial resolution (res_lon, res_lat) centered at each vertex (or face) of the provided icosphere. Sample locations are given by spherical coordinates

    icosphere: icosphere object
    Kh: scalar height of planar kernel
    Kw: scalar width of planar kernel
    res_lat: scalar latitude resolution of kernel
    res_lon: scalar longitude resolution of kernel
    source: {'face' or 'vertex'}

    returns 1 x {F,V} x kh*kw x 2 sampling map per mesh element in spherical coords
    '''

    spherical_coords = convert_3d_to_spherical(points)
    S = spherical_coords
    return gnomonic_kernel(S, kh, kw, res_lat, res_lon)

def get_tangent_images(image_path, scale_factor, base_order, sample_order, points, num_samples):

    img = load_torch_img(image_path)[:3, ...].float() # inputs/I1.png
    img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)

    tex_image = tangent_images(img, base_order, sample_order, points, num_samples).byte()


    for i in range(tex_image.shape[1]):
        img = tex_image[:, i, ...]

        I2 = torch2numpy(img.byte())

    return tex_image

def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('path_o') # test: inputs/I1.png
    args = parser.parse_args()

    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------
    base_order = 1  # Base sphere resolution
    scale_factor = 1  # How much to scale input equirectangular image by

    # ----------------------------------------------
    # Compute necessary data
    # ----------------------------------------------
    # 80 baricenter points

    icosphere     = generate_icosphere(base_order)      # controles the sample extension
    points        = icosphere.get_face_barycenters()    # project tangent planes in that points 3D
    num_samples   = 16                                  # resolution of resultan image

    tangent_images = get_tangent_images(args.path_o, scale_factor, base_order, sample_order, points, num_samples)


if __name__ == '__main__':
    main()


