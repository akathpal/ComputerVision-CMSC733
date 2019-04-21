import numpy as np
import cv2
# import matplotlib.pyplot as plt
from findCorrespondance import findCorrespondance
from GetInliersRANSAC import GetInliersRANSAC
# from EstimateFundamentalMatrix import EstimateFundamentalMatrix
# from EstimateFundamentalMatrix3 import *
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix


# elimination_threshold = 5
Show_correspondence = True

all_Fs = []
all_inliers = []

n_images = 6
Mx = []
My = []
M = []
for i in range(1, n_images):
    mx = []
    my = []
    m = []
    for j in range(i + 1, n_images + 1):
        if (i != j):
            print("\n ---------- \n\n Finding Correspondance between image "
                 + str(i)
                + " and " + str(j) + ":")
        x_list, y_list, binary_list, rgb_list = findCorrespondance(i, j)
#         Tracer()()
        if(j == i + 1):
            mx = x_list
            my = y_list
            m = binary_list
        else:
            mx = np.hstack((mx, x_list[:, 1].reshape((-1, 1))))
            my = np.hstack((my, y_list[:, 1].reshape((-1, 1))))
            m = np.hstack((m, binary_list[:, 1].reshape((-1, 1))))
#     Tracer()()
    if(i == 1):
        Mx = mx
        My = my
        M = m
    else:

        mx = np.hstack((np.zeros((mx.shape[0], i - 1)), mx))
        my = np.hstack((np.zeros((my.shape[0], i - 1)), my))
        m = np.hstack((np.zeros((m.shape[0], i - 1)), m))
        assert Mx.shape[1] == mx.shape[1], "SHape not matched"
        Mx = np.vstack((Mx, mx))
        assert My.shape[1] == my.shape[1], "Shape My not matched"
        My = np.vstack((My, my))
        M = np.vstack((M, m))

img1 = 1
img2 = 2

output = np.logical_and(M[:, img1-1], M[:, img2-1])
indices, = np.where(output == True)
pts1 = np.hstack((Mx[indices, img1-1].reshape((-1, 1)), My[indices, img1-1].reshape((-1, 1))))
pts2 = np.hstack((Mx[indices, img2-1].reshape((-1, 1)), My[indices, img2-1].reshape((-1, 1))))


best_F, inliers_a, inliers_b = GetInliersRANSAC(np.int32(pts1), np.int32(pts2))
# finalF, inliers = GetInliersRANSAC(correspondence_list, elimination_threshold)
print("Final F = ", best_F)
all_Fs.append(best_F)
all_inliers.append(np.hstack((inliers_a, inliers_b)))
print("Total Number of points found between image " + str(i) + " and "
    + str(j) + " is = " + str(len(pts1)))

print("Number of inliners between image " + str(i) + " and "
    + str(j) + " is = " + str(len(inliers_a)))

if Show_correspondence is True:
    out = DrawCorrespondence(i, j, inliers_a, inliers_b)

    cv2.imshow("img3", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("shape of all_inliers", np.shape(all_inliers))
print("shape of all_Fs", np.shape(all_Fs))
# A = np.array(correspondence_list)
# print(A.shape)
