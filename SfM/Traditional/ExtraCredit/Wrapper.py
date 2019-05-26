import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from GetInliersRANSAC import GetInliersRANSAC
from ExtractCameraPose import ExtractCameraPose
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
import sys

sys.dont_write_bytecode = True

img1 = cv2.imread('../Data/Imgs/1.jpeg')  # queryImage
img2 = cv2.imread('../Data/Imgs/3.jpeg')  # trainImage

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Change to True to detect live. It will still run Sift descriptor and
# will take considerable amount of time
LoadData = True

if (not LoadData):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    print("crossed")
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 600)
    cv2.imshow('image', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    np.save('pts1', pts1)
    np.save('pts2', pts2)

if (LoadData):
    pts1 = np.load('pts1.npy')
    pts2 = np.load('pts2.npy')

_, pts1, pts2 = GetInliersRANSAC(np.float32(pts1), np.float32(pts2))

visualize = False

K = np.array([[3.79728180e+03, 0.00000000e+00, 2.08038461e+03],
              [0.00000000e+00, 3.83458104e+03, 1.76272469e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

best_F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))

if (visualize):
    out = DrawCorrespondence(img1, img2, pts1, pts2)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 600)
    cv2.imshow('image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

E = EssentialMatrixFromFundamentalMatrix(best_F, K)
R_set, C_set = ExtractCameraPose(E, K)
X_set = []

for n in range(0, 4):
    X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), C_set[n].T,
                             R_set[n], np.float32(pts1), np.float32(pts2))
    X_set.append(X1)

X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)

colors = []
for i in range(len(pts1)):
    p = img1[int(pts1[i, 1])][int(pts1[i, 0])]
    colors.append([p[0] / 255.0, p[1] / 255.0, p[2] / 255.0])

plt.scatter(X[:, 0], X[:, 2], c=colors, cmap='viridis', s=4)
ax1 = plt.gca()
ax1.set_xlabel('x')
ax1.set_ylabel('z')

plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=colors, s=4)
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylabel('z')

plt.show()
