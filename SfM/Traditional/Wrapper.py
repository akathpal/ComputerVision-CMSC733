"""Main file to implement SFM project for CMSC 733

Attributes:
    img1 (int): Input Image 1
    img2 (int): Input Image 2
    K (array): Intrinsic Matrix
    n_images (int): Number of Images in Dataset
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import argparse
from LoadData import *
from GetInliersRANSAC import GetInliersRANSAC
from ExtractCameraPose import ExtractCameraPose
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from PnPRANSAC import *
from BundleAdjustment import *
from BuildVisibilityMatrix import *
from NonLinearPnP import *
from DrawCameras import *
import sys

sys.dont_write_bytecode = True

# Camera Intrinsic Matrix
K = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038], [0, 0, 1]])

n_images = 6
img1 = 1
img2 = 4


def main():
    """Main Function for implementing SFM
    """
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        '--DataPath', default="./Data/", help='Folder of Images')
    Parser.add_argument(
        '--Visualize', default=False, help='Show correspondences')
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    visualize = Args.Visualize

    # Create matrices for All correspondences

    Mx, My, M, Color = LoadData(DataPath)

    #  Filter M for inliers
    M, outlier_indices = inlier_filter(Mx, My, M, n_images)

    # Save M matrix in case needed to lower the run time by loading
    # np.save('M', M)
    M = np.load('M.npy')

    recon_bin = np.zeros((M.shape[0], 1))
    X_3D = np.zeros((M.shape[0], 3))

    #  We have all inliers at this point in M
    output = np.logical_and(M[:, img1 - 1], M[:, img2 - 1])
    outlier = np.logical_and(outlier_indices[:, img1 - 1],
                             outlier_indices[:, img2 - 1])
    outlier_idx = np.where(outlier == True)
    indices, = np.where(output == True)
    rgb_list = Color[indices]

    pts1 = np.hstack((Mx[indices, img1 - 1].reshape((-1, 1)),
                      My[indices, img1 - 1].reshape((-1, 1))))
    pts2 = np.hstack((Mx[indices, img2 - 1].reshape((-1, 1)),
                      My[indices, img2 - 1].reshape((-1, 1))))
    best_F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))

    outlier1 = np.hstack((Mx[outlier_idx, img1 - 1].reshape((-1, 1)),
                          My[outlier_idx, img1 - 1].reshape((-1, 1))))
    outlier2 = np.hstack((Mx[outlier_idx, img2 - 1].reshape((-1, 1)),
                          My[outlier_idx, img2 - 1].reshape((-1, 1))))

    if (visualize):
        out = DrawCorrespondence(
            img1, img2, pts1, pts2, outlier1, outlier2, DrawOutlier=False)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000, 600)
        cv2.imshow('image', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    E = EssentialMatrixFromFundamentalMatrix(best_F, K)
    R_set, C_set = ExtractCameraPose(E, K)
    X_set = []
    color = ['r', 'g', 'b', 'k']

    for n in range(0, 4):
        X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3),
                                 C_set[n].T, R_set[n], np.float32(pts1),
                                 np.float32(pts2))
        X_set.append(X1)

    X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)

    # X = NonLinearTriangulation(K,np.float32(pts1),np.float32(pts32),X,np.eye(3),np.zeros((3,1)),R,C)
    recon_bin = np.zeros((M.shape[0], 1))
    X_3D = np.zeros((M.shape[0], 3))
    Visibility = np.zeros((M.shape[0], n_images))

    X = NonLinearTriangulation(K, np.float32(pts1), np.float32(pts2), X,
                               np.eye(3), np.zeros((3, 1)), R, C)

    recon_bin[indices] = 1
    X_3D[indices, :] = X
    Visibility[indices, img1 - 1] = 1
    Visibility[indices, img2 - 1] = 1

    # # Plotting non linear triangulation output
    # plt.scatter(X[:, 0], X[:, 2], c='r', s=4)
    # ax = plt.gca()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')

    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.1, 2])

    # plt.show()

    Cset = []
    Rset = []

    Cset.append(C)
    Rset.append(R)

    r_indx = [img1, img2]

    for i in range(0, n_images):

        if (np.isin(r_indx, i)[0]):
            print(i, "Skiiping this frame")
            continue

        output = np.logical_and(recon_bin, M[:, i].reshape((-1, 1)))
        # print(output.shape)
        indices, _ = np.where(output == True)
        #     print("type indices",type(indices))
        if (len(indices) < 8):
            continue

    #     print("M",My[indices,i].reshape((-1,1)))
        x = np.transpose([Mx[indices, i], My[indices, i]])
        X = X_3D[indices, :]

        C, R = PnPRANSAC(X, x, K)
        # print(C)
        # print(R)

        C, R = NonLinearPnP(X, x, K, C, R)
        # print(C)
        # print(R)

        Cset.append(C)
        Rset.append(R)
        r_indx.append(i)
        Visibility[indices, i] = 1
        for j in range(0, len(r_indx) - 1):
            output = np.logical_and(
                np.logical_and(1 - recon_bin, M[:, r_indx[j]].reshape(
                    (-1, 1))), M[:, i].reshape((-1, 1)))
            indices, _ = np.where(output == True)
            if (len(indices) < 8):
                continue

            x1 = np.hstack((Mx[indices, r_indx[j]].reshape((-1, 1)),
                            My[indices, r_indx[j]].reshape((-1, 1))))
            x2 = np.hstack((Mx[indices, i].reshape((-1, 1)),
                            My[indices, i].reshape((-1, 1))))

            X = LinearTriangulation(K, Cset[j], Rset[j], C, R, x1, x2)

            # X = NonLinearTriangulation(K, x1, x2, X, Rset[j],Cset[j],R,C);
            X_3D[indices, :] = X
            recon_bin[indices] = 1
            Visibility[indices, r_indx[j]] = 1
            Visibility[indices, j] = 1

        for o in range(len(X_3D)):
            if (X_3D[o, 2] < 0):
                Visibility[o, :] = 0
                recon_bin[o] = 0

        V_bundle = BuildVisibilityMatrix(Visibility, r_indx)

        point_indices, _ = np.where(recon_bin == 1)
        camera_indices = i * np.ones((len(point_indices), 1))

        points_2d = np.hstack((Mx[point_indices, i].reshape((-1, 1)),
                               Mx[point_indices, i].reshape((-1, 1))))

        Rset, Cset, X_3D = BundleAdjustment(Cset, Rset, X_3D, K, points_2d,
                                            camera_indices, recon_bin,
                                            V_bundle)

    ind, _ = np.where(recon_bin == 1)
    X_3D = X_3D[ind, :]
    Color = Color[ind, :]
    # For 3D plotting
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    ax.scatter3D(
        X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=Color / 255.0,
        s=1)  # cmap='viridis',
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.5, 1])
    ax.set_zlim([0, 1.5])

    plt.show()

    # For 2D plotting

    plt.scatter(X_3D[:, 0], X_3D[:, 2], c=Color / 255.0, s=1)
    DrawCameras(C_set, R_set)
    ax1 = plt.gca()
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    # ax.set_xlim([-0.5, 1])
    # ax.set_ylim([-0.5, 2])

    plt.show()


if __name__ == '__main__':
    main()
