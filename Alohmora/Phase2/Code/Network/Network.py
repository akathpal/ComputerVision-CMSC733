"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Abhishek Kathpal
M.Eng. Robotics
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
     - logits output of the network
    prSoftMax - softmax output of the network
    """

    # Convolutional Layers     
    x = tf.layers.conv2d(inputs=Img, name='conv1', padding='same',filters=16, kernel_size=5, activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm1')
    x = tf.nn.relu(x, name='relu_1')
    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)
    
    x = tf.layers.conv2d(inputs=x, name='conv2', padding='same',filters=32, kernel_size=5, activation=None)
    x = tf.layers.batch_normalization(x, name='batch_norm2')
    x = tf.nn.relu(x, name='relu_2')
    
    x = tf.layers.conv2d(inputs=x, name='conv3', padding='same',filters=64, kernel_size=5, activation=None)
    x = tf.layers.batch_normalization(x, name='batch_norm3')
    x = tf.nn.relu(x, name='relu_3')
    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    #flattening layer
    x = tf.contrib.layers.flatten(x)

    #Fully-connected layers
    x = tf.layers.dense(inputs=x, name='fc_1',units=512, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc_2',units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc_3',units=64, activation=tf.nn.relu)   
    x = tf.layers.dense(inputs=x, name='fc_final',units=10, activation=None)

    prLogits = x
    prSoftMax = tf.nn.softmax(x)

    return prLogits, prSoftMax

def Global_Average_Pooling(x, stride=1):
    
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter


def residual_blocks(blocks,x,prev_filter, num_filters, kernel_size):

    for i in range(blocks):
        if(prev_filter == num_filters):
            orig = x
        else:
            orig = tf.layers.conv2d(inputs=x, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None, strides = 1)
            orig = tf.layers.batch_normalization(orig)

        x = tf.layers.conv2d(inputs=x, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None, strides = 1)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(inputs=x, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None, strides = 1)
        x = tf.layers.batch_normalization(x)

        x = tf.math.add(x, orig)

        x = tf.nn.relu(x)
    return x

def ResNetModel(Img, ImageSize, MiniBatchSize):
   
    x = tf.layers.conv2d(inputs=Img, padding='same', filters=16, kernel_size=3, activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = residual_blocks(1, x, 16,32,5)
    x = residual_blocks(2, x, 32,64,5)
  
    x = Global_Average_Pooling(x, stride=1)
    x = tf.contrib.layers.flatten(x)

    x = tf.layers.dense(inputs=x, name='fc_1',units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc_2',units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc_final',units=10, activation=None)

    prLogits = x
    prSoftMax = tf.nn.softmax(x)
    return prLogits, prSoftMax

