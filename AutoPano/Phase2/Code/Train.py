#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park


Abhishek Kathpal (akathpal@terpmail.umd.edu) 
M.Eng. Robotics,
University of Maryland, College Park
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
from matplotlib import pyplot as plt
from Misc.TFSpatialTransformer import *
import random
# Don't generate pyc codes
sys.dont_write_bytecode = True


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
	if (ModelType.lower() == 'sup'):
		ImageBatch = []
		LabelBatch = []
		ImageNum = 0
		while ImageNum < MiniBatchSize:
			# Generate random image
			RandIdx = random.randint(0, len(DirNamesTrain)-1)
		
			RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'   
			ImageNum += 1
			patchSize = 128
			r = 16

			img_orig = np.float32(plt.imread(RandImageName))

			if(len(img_orig.shape)==3):
				img = cv2.cvtColor(img_orig,cv2.COLOR_RGB2GRAY)
			else:
				img = img_orig

			img=(img-np.mean(img))/255
	
			if(img.shape[1]-r-patchSize)>r+1 and (img.shape[0]-r-patchSize)>r+1:
				x = np.random.randint(r, img.shape[1]-r-patchSize)  
				y = np.random.randint(r, img.shape[0]-r-patchSize)
			else:
				ImageNum = ImageNum -1
				continue  
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

			# Append All Images and Mask
			ImageBatch.append(imgData)
			LabelBatch.append(hData.reshape(8,))

		return ImageBatch,LabelBatch

	else:

		# print("Unsupervised Approach")
		I1FullBatch = []
		PatchBatch = []
		CornerBatch = []
		I2Batch = []

		ImageNum = 0
		while ImageNum < MiniBatchSize:
			# Generate random image
			RandIdx = random.randint(0, len(DirNamesTrain)-1)

			RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'   
			ImageNum += 1
			patchSize = 128
			r = 32

			img_orig = plt.imread(RandImageName)
			
			img_orig = np.float32(img_orig)

			if(len(img_orig.shape)==3):
				img = cv2.cvtColor(img_orig,cv2.COLOR_RGB2GRAY)
			else:
				img = img_orig

			img=(img-np.mean(img))/255
			img = cv2.resize(img,(320,240))
			# img = cv2.resize(img,(ImageSize[0],ImageSize[1]))
			# print(img.shape[1]-r-patchSize)
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

			# Append All Images and Mask
			I1FullBatch.append(np.float32(img))
			PatchBatch.append(imgData)
			CornerBatch.append(np.float32(src))
			I2Batch.append(np.float32(patch2.reshape(128,128,1)))
	
		return I1FullBatch, PatchBatch, CornerBatch, I2Batch


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

	
def TrainOperation(ImgPH, LabelPH, CornerPH, I2PH, I1FullPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
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
	if ModelType.lower() == 'sup':
		print("Supervised")
		H4pt = Supervised_HomographyModel(ImgPH, ImageSize, MiniBatchSize)

		with tf.name_scope('Loss'):
			loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4pt,LabelPH))))

		with tf.name_scope('Adam'):
			Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
	else:
		print("Unsupervised")
		pred_I2,I2 = Unsupervised_HomographyModel(ImgPH, CornerPH, I2PH, ImageSize, MiniBatchSize)

		with tf.name_scope('Loss'):
			loss = tf.reduce_mean(tf.abs(pred_I2 - I2))


		with tf.name_scope('Adam'):
			Optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

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
				if ModelType.lower() == "sup":
					ImgBatch,LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
					FeedDict = {ImgPH: ImgBatch, LabelPH: LabelBatch}
					
				else:
					I1FullBatch, PatchBatch, CornerBatch, I2Batch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
					FeedDict = {ImgPH: PatchBatch, CornerPH: CornerBatch, I2PH: I2Batch}
					
				_, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
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
			Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
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
	Parser.add_argument('--BasePath', default='../Data', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:1')
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

	patchSize = 128
	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None

	if ModelType.lower() == "sup":
		ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, patchSize, patchSize, 2))
		NumTrainSamples = 20000
	else:
		ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
		
		NumTrainSamples = 5000
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	

	# Define PlaceHolder variables for Input and Predicted output
	
	LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8))
	CornerPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 4,2))
	I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128,1))
	I1FullPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1],ImageSize[2]))


	
   
	TrainOperation(ImgPH, LabelPH,CornerPH, I2PH, I1FullPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
		
	
if __name__ == '__main__':
	main()
 
