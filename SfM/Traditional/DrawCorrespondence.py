""" File to implement function to draw correspondence
"""
import cv2
import numpy as np
import sys

sys.dont_write_bytecode = True


def DrawCorrespondence(i, j, inliers_a, inliers_b, outlier1, outlier2,
                       DrawOutliers):
    """Function to draw correspondences on input image

    Args:
        i (int): Input1 image number
        j (int): Input2 image number
        inliers_a (TYPE): list of inliers in image1
        inliers_b (TYPE): list of inliers in image2
        outlier1 (TYPE): list of outiers in image1
        outlier2 (TYPE): list of outiers in image2
        DrawOutliers (TYPE): Bool to draw outliers

    Returns:
        TYPE: image
    """
    img1 = cv2.imread('Data/' + str(i) + '.jpg')
    img2 = cv2.imread('Data/' + str(j) + '.jpg')

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:cols1 + cols2, :] = img2
    radius = 4
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    thickness = 1

    assert len(inliers_a) == len(inliers_b), "inliers in images not equal"
    for m in range(0, len(inliers_a)):
        # Draw small circle on image 1
        cv2.circle(out, (int(inliers_a[m][0]), int(inliers_a[m][1])), radius,
                   RED, -1)

        # Draw small circle on image 2
        cv2.circle(out, (int(inliers_b[m][0]) + cols1, int(inliers_b[m][1])),
                   radius, GREEN, -1)

        # Draw line connecting circles
        cv2.line(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
                 (int(inliers_b[m][0]) + cols1, int(inliers_b[m][1])), GREEN,
                 thickness)
    if (DrawOutliers):
        assert len(outlier1) == len(outlier2), "outliers in images not equal"
        print("Length of outliers to plot" + str(len(outlier1)))
        for n in range(0, len(outlier1)):
            # Draw small circle on image 1
            cv2.circle(out, (int(outlier1[n][0]), int(outlier1[n][1])), radius,
                       BLUE, -1)

            # Draw small circle on image 2
            cv2.circle(out, (int(outlier2[n][0]) + cols1, int(outlier2[n][1])),
                       radius, BLUE, -1)

            # Draw line connecting circles
            cv2.line(out, (int(outlier1[n][0]), int(outlier1[n][1])),
                     (int(outlier2[n][0]) + cols1, int(outlier2[n][1])), RED,
                     thickness)

    return out
