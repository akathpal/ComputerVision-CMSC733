#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Abhishek Kathpal
University of Maryland,College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import pickle
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import Supervised_HomographyModel,Unsupervised_HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True


def extract(data):
	"""
	Extracting training data and labels from pickle files 
	"""
	f = open(data, 'rb')
	out = pickle.load(f)
	features = np.array(out['features'])
	labels = np.array(out['labels'])
	f.close()
	return features,labels


def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType):
	"""
	Inputs: 
	BasePath - Path to COCO folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	ImageSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels 
	"""

	ImageNum = 0
	I1Batch = []
	LabelBatch = []
	if (ModelType.lower() == 'supervised'):
		print("Supervised_approach")
		features,labels=extract('training.pkl')

		ImageNum = 0
		while ImageNum < MiniBatchSize:
		    # Generate random image
		    NumTrainImages=5000
		    RandIdx = random.randint(0, NumTrainImages-1)       
		    ImageNum += 1
			
			##########################################################
			# Add any standardization or data augmentation here!
			##########################################################
		    I1 = np.float32(features[RandIdx])
		    I1=(I1-np.mean(I1))/255

		    t = labels[RandIdx].reshape((1,8))
		    label = t[0]

		    # Append All Images and Mask
		    I1Batch.append(I1)
		    LabelBatch.append(label)

	else:
		print("Unsupervised Approach")



	    
	return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""      
	# Predict output with forward pass
	if ModelType.lower() == 'supervised':
		H4pt = Supervised_HomographyModel(ImgPH, ImageSize, MiniBatchSize)

		with tf.name_scope('Loss'):
		    ###############################################
		    # Fill your loss function of choice here!
		    ###############################################
		    loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4pt,LabelPH))))

		with tf.name_scope('Adam'):
			###############################################
			# Fill your optimizer of choice here!
			###############################################
			Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
	else:
		H4pt = Unsupervised_HomographyModel(ImgPH, corner_pts, ImageSize, MiniBatchSize)

		with tf.name_scope('Loss'):
		    ###############################################
		    # Fill your loss function of choice here!
		    ###############################################
		    loss = tf.stop_gradient(tf.reduce_mean(tf.abs(pred_I2 - I2)))


		with tf.name_scope('Adam'):
			###############################################
			# Fill your optimizer of choice here!
			###############################################
			Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	EpochLossPH = tf.placeholder(tf.float32, shape=None)
	loss_summary = tf.summary.scalar('LossEveryIter', loss)
	epoch_loss_summary = tf.summary.scalar('LossPerEpoch', EpochLossPH)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operation
	MergedSummaryOP1 = tf.summary.merge([loss_summary])
	MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])
	# MergedSummaryOP = tf.summary.merge_all()

	# Setup Saver
	Saver = tf.train.Saver()
	AccOverEpochs=np.array([0,0])
	with tf.Session() as sess:       
	    if LatestFile is not None:
	        Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
	        # Extract only numbers from the name
	        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
	        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
	    else:
	        sess.run(tf.global_variables_initializer())
	        StartEpoch = 0
	        print('New model initialized....')

	    # Tensorboard
	    Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
	        
	    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
	        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
	        Loss=[]
	        epoch_loss=0
	        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
	            I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
	            FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
	            _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
	            #print(shapeH4pt,shapeLabel).
	            Loss.append(LossThisBatch)
	            epoch_loss = epoch_loss + LossThisBatch
	            # Save checkpoint every some SaveCheckPoint's iterations
	            if PerEpochCounter % SaveCheckPoint == 0:
	                # Save the Model learnt in this epoch
	                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
	                Saver.save(sess,  save_path=SaveName)
	                print('\n' + SaveName + ' Model Saved...')

	            # Tensorboard
	            Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
	          

	        epoch_loss = epoch_loss/NumIterationsPerEpoch
	        
	        print(np.mean(Loss))
	        # Save model every epoch
	        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
	        Saver.save(sess, save_path=SaveName)
	        print('\n' + SaveName + ' Model Saved...')
	        Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={epoch_loss_summary: epoch_loss})
	        Writer.add_summary(Summary_epoch,Epochs)
	        Writer.flush()
            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/media/nitin/Research/Homing/SpectralCompression/COCO', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='supervised', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=2, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    print("here")

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8)) # OneHOT labels
    
    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, 5000, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
        
    
if __name__ == '__main__':
    main()
 
