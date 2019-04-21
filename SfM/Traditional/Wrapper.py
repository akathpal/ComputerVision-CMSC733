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

# Camera Intrinsic Matrix
K = np.array([[568.996140852,0,643.21055941],
         [0, 568.988362396, 477.982801038],
         [0, 0, 1]])
img1 = 5
img2 = 6
n_images = 6
limit = 15



def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Data/", help='Folder of Images')
    Parser.add_argument('--Visualize', default=False, help='Show correspondences')
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    visualize = Args.Visualize
    

    Mx,My,M = loadData(DataPath)

    for i in range(1, n_images):
        for j in range(i+1, n_images + 1):    
    
            output = np.logical_and(M[:, img1-1], M[:, img2-1])
            indices, = np.where(output == True)

            if(len(indices)<8):
                continue
            
            pts1 = np.hstack((Mx[indices,img1-1].reshape((-1,1)),My[indices,img1-1].reshape((-1,1))))
            pts2 = np.hstack((Mx[indices,img2-1].reshape((-1,1)),My[indices,img2-1].reshape((-1,1))))
            best_F, inliers_a, inliers_b = GetInliersRANSAC(np.int32(pts1), np.int32(pts2))

            if(visualize):
                out = DrawCorrespondence(img1, img2, inliers_a, inliers_b)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 1000,600)
                cv2.imshow('image', out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            
            E = EssentialMatrixFromFundamentalMatrix(best_F,K)
            R_set,C_set = ExtractCameraPose(E,K)

            X_set = []
            for n in range(0,4):
                X_set.append(LinearTriangulation(K,np.zeros((3,1)),np.identity(3),C_set[n].T,R_set[n],np.int32(pts1),np.int32(pts2)))

            X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)


            plt.scatter(X[:,0], X[:,2], c=X[:,2], cmap='viridis',s = 1)
            axes = plt.gca()
            axes.set_xlim([-limit,limit])
            axes.set_ylim([-limit,limit])
    plt.show()




if __name__ == '__main__':
    main()



