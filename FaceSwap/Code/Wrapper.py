#! /usr/bin/env python

import sys
import numpy as np
import cv2
import imutils
import random
import math


from scipy.interpolate import interp2d
import argparse
import os

from prnet.api import PRN
from prnet.main import prnetSwap

from prnet.api_two import PRN_two
from prnet.main_two import prnetSwap_two

from traditional.facial_landmarks import facial_landmarks
from traditional.main import traditional
from traditional.twoFaces import twoFaces


if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Face Swapping')

	parser.add_argument('--input_path', default="../TestSet/", type=str,help='path to the input')
	parser.add_argument('--face', default='Rambo', type=str,help='path to face')
	parser.add_argument('--video', default='Test1', type=str,help='path to the input video')
	parser.add_argument('--method', default='tps', type=str,help='affine, tri, tps, prnet')
	parser.add_argument('--resize', default=False, type=bool,help='True or False input resizing')
	parser.add_argument('--mode', default=1, type=int,help='1- swap face in video with image, 2- swap two faces within video')



	Args = parser.parse_args()
	video_path = Args.input_path+Args.video+'.mp4'
	mode = Args.mode
	video = Args.video
	method = Args.method
	resize = Args.resize
	w = 320

	cap = cv2.VideoCapture(video_path)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("No. of frames = "+str(length))

	ret,img = cap.read()


	if resize:
	    img = imutils.resize(img,width = w)
	height = img.shape[0]
	width = img.shape[1]

	# Defining Video Writer Object
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	out = cv2.VideoWriter('{}_Output_{}.avi'.format(method,video),fourcc, 15, (width,height))

	count = 0

	

	if(mode==1):

	    face_path = Args.input_path+Args.face+'.jpg'
	    

	    img1 = cv2.imread(face_path);
	    if resize:
	        img1 = imutils.resize(img1,width = w)
	    
	    faces_num,points1 = facial_landmarks(img1)
	    if(faces_num!=1):
	        print("More than 1 or zero face detected...Exiting")
	        exit()
	      
	    ret,img2 = cap.read()
	    if resize:
	        img2 = imutils.resize(img2,width = w)
	    height = img2.shape[0]
	    width = img2.shape[1]

	    
	    if(method=="prnet"):
		    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		    prn = PRN(is_dlib = True)

	    while(cap.isOpened()):

	        
	        count += 1

	        ret,img2 = cap.read()
	        

	        if(ret==True):

	            # img2 = imutils.rotate(img2,90)
	            if resize:
	                img2 = imutils.resize(img2,width = w)


	            if(method=="prnet"):

	                pos,output = prnetSwap(prn,img2,img1)
	                
	                if pos is None:
	                    continue
	                else:
	                    print("Frame"+str(count))

	            else:

	                faces_num,points2 = facial_landmarks(img2)
	                if(faces_num==0):
	                    continue
	                else:
	                    print("Frame"+str(count))

	                output = traditional(img1,img2,points1,points2,method)

	                
	            
	            cv2.imshow("Face Swapped", output)
	            cv2.waitKey(100)
	            out.write(output)
	            
	            if cv2.waitKey(1) & 0xff==ord('q'):
	                cv2.destroyAllWindows()
	                break
	        else:
	            exit()

	else:
		
		print("Mode "+str(mode))
		
		

		if(method=="prnet"):
		    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		    prn = PRN_two(is_dlib = True)

		while(cap.isOpened()):

		    
			count += 1
			ret,img = cap.read()
			
			if(ret==True):

			    img = imutils.rotate(img,180)
			    if resize:
			        img = imutils.resize(img,width = w)


			    if(method=="prnet"):

			        pos,output = prnetSwap_two(prn,img,img)
			        
			        if pos is None:
			            continue
			        else:
			            print("Frame"+str(count))

			    else:

			        faces_num,points = twoFaces(img)

			        if(faces_num!=2):
						print("{} faces detected in frame {}".format(faces_num,count))
						continue
			        else:
						points1 = points[0]
						points2 = points[1]

						print("Frame"+str(count))
						# print(len(points1))
						# print(len(points2))

			        temp = traditional(img,img,points1,points2,method)
			        output = traditional(img,temp,points2,points1,method)
			        # print(output.shape)

			        
			    
			    cv2.imshow("Face Swapped", output)
			    cv2.waitKey(100)
			    out.write(output)
			    
			    if cv2.waitKey(1) & 0xff==ord('q'):
			        cv2.destroyAllWindows()
			        break
			else:
			    exit()

        
