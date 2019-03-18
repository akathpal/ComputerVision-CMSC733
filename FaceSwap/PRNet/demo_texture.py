import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from api import PRN
from utils.render import render_texture
import cv2


def texture_editing(prn, args):
    # read image
    image = imread(args.image_path)
    [h, w, _] = image.shape

    #-- 1. 3d reconstruction -> get texture. 
    pos = prn.process(image) 
    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
  
    ref_image = imread(args.ref_path)
    ref_pos = prn.process(ref_image)
    ref_image = ref_image/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture#(texture + ref_texture)/2.

    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)
    
    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    new_image = image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
   
    # save output
    imsave(args.output_path, output) 
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    parser.add_argument('-i', '--image_path', default='TestImages/AFLW2000/image00081.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-r', '--ref_path', default='TestImages/trump.jpg', type=str, 
                        help='path to reference image(texture ref)')
    parser.add_argument('-o', '--output_path', default='TestImages/output.jpg', type=str, 
                        help='path to save output')
    parser.add_argument('--gpu', default='0', type=str, 
                        help='set gpu id, -1 for CPU')

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = True) 

    texture_editing(prn, parser.parse_args())