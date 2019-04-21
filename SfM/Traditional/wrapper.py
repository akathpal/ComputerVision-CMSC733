import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from LoadData import *
from GetInliersRANSAC import GetInliersRANSAC
from ExtractCameraPose import ExtractCameraPose
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *


if __name__ == '__main__':
    main()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Data/", help='Folder of Images')
    Parser.add_argument('--Visualize', default=False, help='Show correspondences')
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    visualize = Args.visualize

    Mx,My,M = loadData(DataPath)

    img1 = 1
    img2 = 4
    n_images = 6
    visualize = False

    output = np.logical_and(M[:, img1-1], M[:, img2-1])
    indices, = np.where(output == True)
    
    pts1 = np.hstack((Mx[indices,img1-1].reshape((-1,1)),My[indices,img1-1].reshape((-1,1))))
    pts2 = np.hstack((Mx[indices,img2-1].reshape((-1,1)),My[indices,img2-1].reshape((-1,1))))
    best_F, inliers_a, inliers_b = GetInliersRANSAC(np.int32(pts1), np.int32(pts2))

    if visualize is True:
        out = DrawCorrespondence(img1, img2, inliers_a, inliers_b)

        cv2.imshow("img3", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    K = np.array([[568.996140852,0,643.21055941],
         [0, 568.988362396, 477.982801038],
         [0, 0, 1]])
    E = EssentialMatrixFromFundamentalMatrix(best_F,K)
    R_set,C_set = ExtractCameraPose(E,K)

    X_set = []
    for n in range(0,4):
        X_set.append(LinearTriangulation(K,np.zeros((3,1)),np.identity(3),C_set[n].T,R_set[n],np.int32(pts1),np.int32(pts2)))

    X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)


    plt.scatter(X_new[:,0], X_new[:,2], c=X_new[:,2], cmap='viridis',s = 1);  #viridis
    plt.set_xlabel('x')
    plt.set_ylabel('z');
    axes = plt.gca()
    axes.set_xlim([-5,5])
    axes.set_ylim([-5,5])



