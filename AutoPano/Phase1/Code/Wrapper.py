#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Abhishek Kathpal (akathpal@terpmail.umd.edu) 
M.Eng. Robotics,
University of Maryland, College Park
"""

# Python libraries 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage.feature import peak_local_max
import random 
from glob import glob

def harris_corner(img,Feature="mineig"):
    img_temp = img.copy()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if Feature == "mineig":
        Rs = cv2.cornerMinEigenVal(gray_img,2,3)
    else:
        Rs = cv2.cornerHarris(gray_img,2,3,k=0.04)

    
    t = Rs<0.01*Rs.max()
    m = Rs>0.01*Rs.max()
    Rs[t] = 0
    corners = np.where(m)
    img_temp[corners]=[255,0,0]
    
    return img_temp,Rs,corners

def anms(Cmap,corners,image,n_best=100):
    img = image.copy()
    C = Cmap.copy()
    locmax = peak_local_max(C,min_distance=10)

    n_strong = locmax.shape[0]
    print(n_strong)

    r = [np.Infinity for i in range(n_strong)]
    x=np.zeros((n_strong,1))
    y=np.zeros((n_strong,1))
    ed=0
    for i in range(n_strong):
        for j in range(n_strong):
            if(C[locmax[j][0],locmax[j][1]] > C[locmax[i][0],locmax[i][1]]):
                ed = (locmax[j][0]-locmax[i][0])**2 + (locmax[j][1]-locmax[i][1])**2
            if ed<r[i]:
                r[i] = ed
                x[i] = locmax[i][0]
                y[i] = locmax[i][1]

    ind = np.argsort(r)
    ind = ind[-n_best:]
    x_best=np.zeros((n_best,1))
    y_best=np.zeros((n_best,1))
    for i in range(n_best):
        x_best[i] = np.int0(x[ind[i]])
        y_best[i] = np.int0(y[ind[i]]) 
        cv2.circle(img,(y_best[i],x_best[i]),3,255,-1)
           
    return x_best,y_best,img

def corner_detect_with_ANMS(img,n=500):   
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img,n,0.01,10)
    corners = np.int0(corners)
    img_temp = img.copy()
    x_best = []
    y_best = []
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_temp,(x,y),3,255,-1)
        x_best.append(y)
        y_best.append(x)
    return x_best,y_best,img_temp,corners

def feature_des(img,x_best,y_best):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    top = 40
    bottom = top
    left = 40
    right = left 
    dst = cv2.copyMakeBorder(gray_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [0])
    features = []
    for i in range(len(x_best)):
        patch = dst[x_best[i]+20:x_best[i]+60,y_best[i]+20:y_best[i]+60]
        patch = cv2.GaussianBlur(patch,(5,5),cv2.BORDER_DEFAULT)
        patch = cv2.resize(patch, (8,8), interpolation = cv2.INTER_AREA)
        feature = np.reshape(patch, (64, ))
        mean = np.mean(feature)
        std = np.std(feature)
        feature = (feature - mean)/std
        features.append(feature)
        # plt.figure(figsize=(15,15))
        # plt.imshow(patch,cmap='gray')
        # plt.show()

    return features,patch

def feature_match(features1,features2,x1,y1,x2,y2,img1,img2):
    
    features1 = np.array(features1,dtype='float32')
    features2 = np.array(features2,dtype='float32')
    bf = cv2.BFMatcher()
    rawMatches = bf.knnMatch(features1,features2,2)

    matches = []
    good = []
    c=0
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            good.append([m[0]])
            c = c+1

    corners1 = []
    corners2 = []
    for i in range(len(x1)):
        corners1.append([x1[i],y1[i]])

    for i in range(len(x2)):
        corners2.append([x2[i],y2[i]])

    pts1 = np.float32(corners1)
    pts2 = np.float32(corners2)
    pts1 = np.reshape(pts1,(len(x1),2))
    pts2 = np.reshape(pts2,(len(x2),2))

    kp1=[]
    kp2=[]
    for i in range(len(pts1)):
        kp1.append(cv2.KeyPoint(pts1[i][1], pts1[i][0], 5))
        kp2.append(cv2.KeyPoint(pts2[i][1], pts2[i][0], 5))

    img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    pts1f = []
    pts2f = []
    pts1t = []
    pts2t = []

    for mat in good:
        pts1f.append([kp1[mat[0].queryIdx].pt[1],kp1[mat[0].queryIdx].pt[0]])
        pts2f.append([kp2[mat[0].trainIdx].pt[1],kp2[mat[0].trainIdx].pt[0]])
        pts1t.append([kp1[mat[0].queryIdx].pt[0],kp1[mat[0].queryIdx].pt[1]])
        pts2t.append([kp2[mat[0].trainIdx].pt[0],kp2[mat[0].trainIdx].pt[1]])

    pts1f = np.float32(np.reshape(pts1f,(len(pts1f),2)))
    pts2f = np.float32(np.reshape(pts2f,(len(pts2f),2)))
    pts1t = np.float32(np.reshape(pts1t,(len(pts1t),2)))
    pts2t = np.float32(np.reshape(pts2t,(len(pts2t),2)))
    
    # plt.figure(figsize=(15,15))
    # plt.imshow(img)
    # plt.show()

    return pts1f,pts2f,pts1t,pts2t

def ransac(pts1, pts2, N=100, t=0.9, thresh=30):

    H_new = np.zeros((3,3))
    max_inliers = 0
    print(len(pts1))
    for j in range(N):

        index = []
        pts = [np.random.randint(0,len(pts1)) for i in range(4)]
        p1 = pts1[pts]
        p2 = pts2[pts]
        H = cv2.getPerspectiveTransform( np.float32(p1), np.float32(p2) )
        inLiers = 0
        for ind in range(len(pts1)):
            source = pts1[ind]
            target = np.array([pts2[ind][0],pts2[ind][1]])
            predict = np.dot(H, np.array([source[0],source[1],1]))
            if predict[2] != 0:
                predict_x = predict[0]/predict[2]
                predict_y = predict[1]/predict[2]
            else:
                predict_x = predict[0]/0.000001
                predict_y = predict[1]/0.000001

            predict = np.array([predict_x,predict_y])
            predict = np.float32([point for point in predict])

            a = np.linalg.norm(target-predict)
            # e = (a - np.mean(a)) / np.std(a)
            if a < thresh:
                inLiers += 1
                index.append(ind)

        if max_inliers < inLiers:
            max_inliers = inLiers
            H_new = H
            if inLiers > t*len(pts1):
                break
    return H_new,index

## 3rd party code for blending images
def warpTwoImages(img1, img2, H):

    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

    return result


def autopano(img1,img2,Feature="mineig"):

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    img1 = cv2.GaussianBlur(img1,(3,3),0)
    img2 = cv2.GaussianBlur(img2,(3,3),0)

    if Feature == "harris" or Feature == "mineig":
        img1_corner,map1,corners1 = harris_corner(img1,Feature)
        img2_corner,map2,corners2 = harris_corner(img2,Feature)

        locmax1 = peak_local_max(map1,min_distance=10)
        n_strong1 = locmax1.shape[0]
        locmax2 = peak_local_max(map2,min_distance=10)
        n_strong2 = locmax2.shape[0]
        n_best = min(n_strong1,n_strong2)

        # plt.figure(figsize=(15,15))
        # plt.subplot(121)
        # plt.axis('off')
        # plt.imshow(img1_corner)
        # plt.subplot(122)
        # plt.axis('off')
        # plt.imshow(img2_corner)
        # plt.show()

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        num = min(n_best,100)
        print("Number of best corners: "+str(num))
        x1,y1,img1_corner = anms(map1,corners1,img1,num)
        x2,y2,img2_corner = anms(map2,corners2,img2,num)

        x1 = np.int0(x1.reshape(1,len(x1)))[0]
        x2 = np.int0(x2.reshape(1,len(x2)))[0]
        y1 = np.int0(y1.reshape(1,len(y1)))[0]
        y2 = np.int0(y2.reshape(1,len(y2)))[0]

    else:
        print("Good Features to track")
        x1,y1,img1_corner,corners1 = corner_detect_with_ANMS(img1)
        x2,y2,img2_corner,corners2 = corner_detect_with_ANMS(img2)


    

    # plt.figure(figsize=(15,15))
    # plt.subplot(121)
    # plt.axis('off')
    # plt.imshow(img1_corner)
    # plt.subplot(122)
    # plt.axis('off')
    # plt.imshow(img2_corner)
    # plt.show()

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    features1,patch1 = feature_des(img1,x1,y1)
    features2,patch2 = feature_des(img2,x2,y2)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    pts1f,pts2f,pts1t,pts2t = feature_match(features1,features2,x1,y1,x2,y2,img1,img2)

    """
    Refine: RANSAC, Estimate Homography
    """
    # print(len(pts1t))
    H,index = ransac(pts1t,pts2t)
    # print(len(index))
    pts1n = []
    pts2n = []
    x1n=[]
    x2n=[]
    y1n=[]
    y2n=[]
    for i in index:
        pts1n.append(pts1f[i])
        x1n.append(np.int0(pts1f[i][0]))
        y1n.append(np.int0(pts1f[i][1]))
        pts2n.append(pts2f[i])
        x2n.append(np.int0(pts2f[i][0]))
        y2n.append(np.int0(pts2f[i][1]))


       
    H = np.float64([pt for pt in H])
    features1,patch1 = feature_des(img1,x1n,y1n)
    features2,patch2 = feature_des(img2,x2n,y2n)
    _,_,_,_ = feature_match(features1,features2,x1n,y1n,x2n,y2n,img1,img2)
    result = cv2.warpPerspective(img1, H,(img1.shape[1], img1.shape[0]))
    # plt.figure(figsize=(15,15))
    # plt.autoscale(True)
    # plt.imshow(result)
    # plt.show()

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    result = warpTwoImages(img2, img1, H)
    plt.figure()
    plt.imshow(result)
    plt.show()
    cv2.imwrite('Final_Output'+'.png',result)

    return result

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default="../Data/Train/Set1", help='Folder of Test Images')
    Parser.add_argument('--Features',default="good",help='good or harris or mineig')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    Feature = Args.Features

    path = str(BasePath) + str("/*.jpg")

    """
    Read a set of images for Panorama stitching
    """
    
    ind = sorted(glob(path))
    
    img = []
    for i in ind:
        img.append(plt.imread(i))

    
    # res = []
    # for i in range(0,len(img)-1,2):
    #     res.append(autopano(img[i],img[i+1],Feature))
    # if len(img)%2 != 0:
    #     res.append(img[len(img)-1])
    # for i in range(len(res)-1):
    #     res[i+1] = autopano(res[i],res[i+1])

    for i in range(0,len(img)-1):
        img[i+1] = autopano(img[i],img[i+1],Feature)
    # res = autopano(img1,img2)
    # final = autopano(img2,img3)
    # result = autopano(res, img3)
    # plt.figure()
    # plt.imshow(result)
    # plt.show()

    
if __name__ == '__main__':
    main()
 
