import cv2
import numpy as np
import sys

sys.dont_write_bytecode = True

def DrawCorrespondence(img1, img2, inliers_a, inliers_b):
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:cols1 + cols2, :] = img2
    radius = 6
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    thickness = 2

    for m in range(0, len(inliers_a)):
        # Draw small circle on image 1
        cv2.circle(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
            radius, RED, thickness)

        # Draw small circle on image 2
        cv2.circle(out, (int(inliers_b[m][0])
            + cols1, int(inliers_b[m][1])), radius, RED, thickness)

        # Draw line connecting circles
        cv2.line(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
            (int(inliers_b[m][0])+cols1, int(inliers_b[m][1])), GREEN, thickness)

    return out
