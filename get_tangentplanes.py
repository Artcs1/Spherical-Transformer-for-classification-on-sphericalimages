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

    points = points#.cuda()
    spherical_coords = convert_3d_to_spherical(points)
    S = spherical_coords
    return gnomonic_kernel(S, kh, kw, res_lat, res_lon)


def gnomonic_kernel2(spherical_coords, kh, kw, res_lat, res_lon):
    lon = spherical_coords[..., 0]
    lat = spherical_coords[..., 1]
    num_samples = spherical_coords.shape[0]

    # Kernel
    x = torch.zeros(kh * kw, device = spherical_coords.device)
    y = torch.zeros(kh * kw, device = spherical_coords.device)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)
            # Project the sphere onto the tangent plane
            x[i * kw + j] = cur_j * res_lon
            y[i * kw + j] = cur_i * res_lat

    # Center the kernel if dimensions are even
    if kh % 2 == 0:
        y += res_lat / 2
    if kw % 2 == 0:
        x += res_lon / 2

    # Equalize views
    lat = lat.view(1, num_samples, 1)
    lon = lon.view(1, num_samples, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    rho = (x**2 + y**2).sqrt()
    nu = rho.atan()
    out_lat = (nu.cos() * lat.sin() + y * nu.sin() * lat.cos() / rho).asin()
    out_lon = lon + torch.atan2(
        x * nu.sin(),
        rho * lat.cos() * nu.cos() - y * lat.sin() * nu.sin())

    # If kernel has an odd-valued dimension, handle the 0 case which resolves to NaN above
    if kh % 2 == 1:
        out_lat[..., [(kh // 2) * kw + kw // 2]] = lat
    if kw % 2 == 1:
        out_lon[..., [(kh // 2) * kw + kw // 2]] = lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return (1, num_samples, kh*kw, 2) map at locations given by <spherical_coords>
    return torch.stack((out_lon, out_lat), -1)

def get_tangent_images(image_path, scale_factor, base_order, points, num_samples, sample_order = 7):

    img = load_torch_img(image_path)[:3, ...].float() # inputs/I1.png
    img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)

    tex_image = tangent_images(img, base_order, sample_order, points, num_samples).byte()


    #for i in range(tex_image.shape[1]):
    #    img = tex_image[:, i, ...]

    #   I2 = torch2numpy(img.byte())

    return tex_image


def get_tangent_images2(img, base_order, points, num_samples, sample_order = 7):
    tex_image = tangent_images(img, base_order, sample_order, points, num_samples).byte()

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

    tangent_images = get_tangent_images(args.path_o, scale_factor, base_order, points, num_samples)


if __name__ == '__main__':
    main()

