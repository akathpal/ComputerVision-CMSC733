#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Abhishek Kathpal
M.Eng. Robotics
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import Supervised_HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def image_data(BasePath):
    RandIdx = random.randint(1, 1000)
    # RandIdx = 1

    RandImageName = BasePath + str(RandIdx) + '.jpg'   
    patchSize = 128
    r = 16

    img_orig = plt.imread(RandImageName)
    

    if(len(img_orig.shape)==3):
        img = cv2.cvtColor(img_orig,cv2.COLOR_RGB2GRAY)
    else:
        img = img_orig


    img=(img-np.mean(img))/255

    if(img.shape[1]-r-patchSize)>r+1 and (img.shape[0]-r-patchSize)>r+1:
        x = np.random.randint(r, img.shape[1]-r-patchSize)  
        y = np.random.randint(r, img.shape[0]-r-patchSize)
    # print(x)

    p1 = (x,y)
    p2 = (patchSize+x, y)
    p3 = (patchSize+x, patchSize+y)
    p4 = (x, patchSize+y)
    src = [p1, p2, p3, p4]
    src = np.array(src)
    dst = []
    for pt in src:
        dst.append((pt[0]+np.random.randint(-r, r), pt[1]+np.random.randint(-r, r)))

    

    H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    H_inv = np.linalg.inv(H)
    warpImg = cv2.warpPerspective(img, H_inv, (img.shape[1],img.shape[0]))

    patch1 = img[y:y + patchSize, x:x + patchSize]
    patch2 = warpImg[y:y + patchSize, x:x + patchSize]

    imgData = np.dstack((patch1, patch2))
    hData = np.subtract(np.array(dst), np.array(src))
    
    return imgData,hData,np.array(src),np.array(dst),img_orig

            

def TestOperation(ImgPH, ImageSize, ModelPath, BasePath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    imgData,hData,src,dst,img = image_data(BasePath)

    H4pt = Supervised_HomographyModel(ImgPH, ImageSize, 1)
    Saver = tf.train.Saver()
  
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
    
        imgData=np.array(imgData).reshape(1,128,128,2)
        
        FeedDict = {ImgPH: imgData}
        Predicted = sess.run(H4pt,FeedDict)
        
    src_new=src+Predicted.reshape(4,2)
    H4pt_new=dst-src_new
    
    cv2.polylines(img,np.int32([src]),True,(0,255,0), 3)
    cv2.polylines(img,np.int32([dst]),True,(255,0,0), 5)
    cv2.polylines(img,np.int32([src_new]),True,(0,0,255), 5)
    plt.figure()
    plt.imshow(img)
    plt.show()
    cv2.imwrite('Final_Output'+'.png',img)



        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/49a0model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data/Val/', help='Path to load images from, Default:BasePath')
    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath

    ImageSize = [1,128,128,2]
 
    # Define PlaceHolder variables for Input and Predicted output
    ImgPH=tf.placeholder(tf.float32, shape=(1, 128, 128, 2))

    TestOperation(ImgPH, ImageSize, ModelPath, BasePath)

     
if __name__ == '__main__':
    main()
 
