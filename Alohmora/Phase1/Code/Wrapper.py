#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Abhishek Kathpal(akathpal@terpmail.umd.edu)
M.Eng. Student, Robotics
University of Maryland, College Park

"""

# Code starts here:

import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.transform import rotate
import glob 
import os
from sklearn.cluster import KMeans

# Helper Functions 

def gauss2d(n,sigma):
    size = int((n-1)/2)
    var = sigma**2
    m = np.asarray([[x**2 + y**2 for x in range(-size,size+1)] for y in range(-size,size+1)])
    output = (1/np.sqrt(2*np.pi*var))*np.exp(-m/(2*var))
    return output

def gauss1d(sigma, mean,x, order):
    x = np.array(x) - mean
    var = sigma**2
    g = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x*x)/(2*var)))
    
    if order == 0:
        return g
    elif order == 1:
        g = -g*((x)/(var))
        return g
    else:
        g = g*(((x*x) - var)/(var**2))
        return g

def log2d(n,sigma):
    size = int((n-1)/2)
    var = sigma**2
    m = np.asarray([[x**2 + y**2 for x in range(-size,size+1)] for y in range(-size,size+1)])
    n = (1/np.sqrt(2*np.pi*var))*np.exp(-m/(2*var))
    output = n*(m - var)/(var**2)
    return output

def binary(img,bin_value):
    binary_img = img * 0
    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            if img[r, c]==bin_value:
                binary_img[r, c] = 1   
            else:
                binary_img[r, c] = 0
    return binary_img

   
def gradient(maps,numbins,mask_l,mask_r):
    gradient = np.zeros((maps.shape[0],maps.shape[1],12))
    for m in range(0,12):
        chi = np.zeros((maps.shape))
        for i in range(1,numbins):
            tmp = binary(maps,i)
            g = convolve2d(tmp, mask_l[m],'same')
            h = convolve2d(tmp, mask_r[m],'same')
            chi = chi + ((g-h)**2) / (g+h+0.0001)
        gradient[:,:,m] = chi
    return gradient

def makefilter(scale, phasex, phasey, pts, sup):
    gx = gauss1d(3*scale, 0,pts[0,...], phasex)
    gy = gauss1d(scale, 0,pts[1,...], phasey)
    image = gx*gy
    image = np.reshape(image,(sup,sup))
    return image

## Defining DoG Filter
def DOG_filter_bank(sigma, orientation):
    sobel = np.asarray([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])

    ims = []
    s = len(sigma)
    o = orientation.size
    # sigma = [1,3]
    # orientation = np.arange(0, 360,360/o)
    plt.figure(figsize=(15,2))
    for i in range(0,s):
        filter_ = convolve2d(gauss2d(5,sigma[i]), sobel)
        for j in range(0,o):
            filt = rotate(filter_,orientation[j])
            ims.append(filt)
            plt.subplot(s,o,o*(i)+j+1)
            plt.axis('off')
            plt.imshow(ims[o*(i)+j],cmap='gray')
            
    plt.show()
    
    return ims

## Defining Gabor filter
def genGabor(sz, omega, theta):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega**2 / (4*np.pi * np.pi**2) * np.exp(- omega**2 / (8*np.pi**2) * ( 4 * x1**2 + y1**2))

    sinusoid = np.cos(omega * x1) * np.exp(np.pi**2 / 2)

    gabor = gauss * sinusoid
    return gabor

#Gabor Filter Bank
def Gabor_filter_bank():
	theta = np.arange(0, np.pi, np.pi/4) # range of theta
	omega = np.arange(0.2, 0.6, 0.1) # range of omega
	params = [(t,o) for o in omega for t in theta]
	FilterBank = []
	gaborParams = []
	for (theta, omega) in params:
	    gaborParam = {'omega':omega, 'theta':theta, 'sz':(128, 128)}
	    Gabor = genGabor((128,128),omega,theta)
	    FilterBank.append(Gabor)
	    gaborParams.append(gaborParam)

	plt.figure()
	n = len(FilterBank)
	for i in range(n):
	    plt.subplot(4,4,i+1)
	    plt.axis('off'); plt.imshow(FilterBank[i],cmap='gray')
	plt.show()
	return FilterBank

#Leung Malik Filter Bank
def LM_filter_bank():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)
    
    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c,-s],[s,c]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale],0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale],0, 2, rotpts, sup)
            count = count + 1
            
    count = nbar+nedge
    
    scales = np.sqrt(2) * np.array([1,2,3,4])
    
    for i in range(len(scales)):
        F[:,:,count]   = gauss2d(sup, scales[i])
        count = count + 1
        
    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1
        
    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1
        
    return F

def half_disk(radius):
    hd = np.zeros((radius*2,radius*2))
    rs = radius**2;
    for i in range(0,radius):
        m = (i-radius)**2
        for j in range(0,2*radius):
            if m+(j-radius)**2 < rs:
                hd[i,j] =1
    return hd

def plot(image,cmap=None):
	plt.imshow(image,cmap)
	plt.axis('off')
	plt.show()
def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	sigma = [1,3]
	orientation = np.arange(0, 360,360/16)
	DoG_filters = DOG_filter_bank(sigma,orientation)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	F = LM_filter_bank()
	print F.shape
	plt.figure(figsize=(6,8))
	for i in range(0,48):
	    plt.subplot(9,6,i+1)
	    plt.axis('off')
	    plt.imshow(F[:,:,i], cmap = 'gray')
	plt.show()
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	Gabor_filters = Gabor_filter_bank()

	filter_bank = []
	count = 0
	for i in range(0,len(DoG_filters)):
	    filter_bank.append(DoG_filters[i])
	    count = count+1

	for i in range(0,48):
	    filter_bank.append(F[:,:,i])
	    count = count+1

	for i in range(len(Gabor_filters)):
	    filter_bank.append(Gabor_filters[i])
	    count = count+1
	    

	"""Half Disc Maps
	"""
	orients = np.arange(0,360,360/4)
	print(orients)
	radii = np.asarray([5,10,15])
	mask_l = []
	mask_r = []
	r = radii.size
	o = orients.size
	print(r,o)
	for i in range(0,radii.size):
	    hd = half_disk(radii[i])
	    for m in range(0,orients.size):
	        mask_1 = rotate(hd,orients[m])
	        mask_l.append(mask_1)
	        mask_2 = rotate(mask_1,180)
	        mask_r.append(mask_2)
	        
	        plt.subplot(r*2,o,o*2*(i)+m+1)
	        plt.axis('off')
	        plt.imshow(mask_1,cmap='gray')
	        plt.subplot(r*2,o,o*2*(i)+m+1+o)
	        plt.axis('off')
	        plt.imshow(mask_2,cmap='gray')
	plt.show()

	os.chdir("../BSDS500/Images")
	cwd = os.getcwd()
	print(cwd)
	cv_img = []
	for img in sorted(glob.glob("*.jpg")):
	    n= cv2.imread(img)
	    cv_img.append(n)
	os.chdir("../../Code")

	img = 7
	plot(cv2.cvtColor(cv_img[img],cv2.COLOR_BGR2RGB))
	

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	tmap=cv2.cvtColor(cv_img[img],cv2.COLOR_BGR2GRAY)
	data = np.zeros((tmap.size,len(filter_bank)))
	for i in range(0,len(filter_bank)):
	    temp_im = cv2.filter2D(tmap, -1, filter_bank[i])
	    temp_im = temp_im.reshape((1,tmap.size))
	    data[:,i] = temp_im


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	k_means = KMeans(n_clusters=64, n_init=4)
	k_means.fit(data)
	labels = k_means.labels_
	tmap = np.reshape(labels,(tmap.shape))

	plot(tmap)

	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	tg = gradient(tmap,64,mask_l,mask_r)
	tgm = np.mean(tg, axis=2);
	plot(tgm)

	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	n = cv2.cvtColor(cv_img[img],cv2.COLOR_BGR2GRAY)
	m = n.reshape((n.shape[0]*n.shape[1]),1)
	k_means = KMeans(n_clusters=16, random_state=4)
	k_means.fit(m)
	labels = k_means.labels_
	bmap = np.reshape(labels,(n.shape[0],n.shape[1]))
	low = np.min(bmap)
	high = np.max(bmap)
	bmap_f = 255*(bmap-low)/np.float((high-low))
	
	plot(bmap_f,'gray')
	

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	bg = gradient(bmap,16,mask_l,mask_r)
	bgm = np.mean(bg, axis=2)
	plot(bgm,'gray')
	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	n = cv_img[img]
	m = n.reshape((n.shape[0]*n.shape[1]),3)
	k_means = KMeans(n_clusters=16, random_state=4)
	k_means.fit(m)
	labels = k_means.labels_
	cmap = np.reshape(labels,(n.shape[0],n.shape[1]))
	plot(cmap)

	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	cg = gradient(cmap,16,mask_l,mask_r)
	cgm = np.mean(cg, axis=2);
	plot(cgm)
	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	os.chdir("../BSDS500/CannyBaseline")
	cwd = os.getcwd()
	print(cwd)
	c = cv2.imread(str(img)+".png")
	plot(c)
	os.chdir("../../Code")

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	os.chdir("../BSDS500/SobelBaseline")
	s = cv2.imread(str(img)+".png")
	plot(s)
	os.chdir("../../Code")

	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	sm = cv2.cvtColor(s,cv2.COLOR_BGR2GRAY)
	cm = cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
	w = 0.5
	pb = ((tgm+bgm+cgm)/3)*(w*cm+(1-w)*sm)
	plot(pb,'gray')
	plt.imsave("PbLite"+str(img)+".png",pb,cmap='gray')
    
if __name__ == '__main__':
    main()
 


