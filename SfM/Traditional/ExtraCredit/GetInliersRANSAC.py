""" This file implements the RANSAC algorithm on given matches
"""
import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
import sys

sys.dont_write_bytecode = True


def GetInliersRANSAC(matches_a, matches_b):
    """Function to implement RANSAC

    Args:
        matches_a (np.array(int32)): Matches of image1
        matches_b (np.array(int32)): Matches of image2
        indices (TYPE): Description

    Returns:
        best_F: F matrix
        inliers_a
        inliers_b
    """
    matches_num = matches_a.shape[0]
    Best_count = 0

    for iter in range(500):
        sampled_idx = np.random.randint(0, matches_num, size=8)
        F = EstimateFundamentalMatrix(matches_a[sampled_idx, :],
                                      matches_b[sampled_idx, :])
        in_a = []
        in_b = []
        update = 0
        for i in range(matches_num):
            matches_aa = np.append(matches_a[i, :], 1)
            matches_bb = np.append(matches_b[i, :], 1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < 0.005:
                in_a.append(matches_a[i, :])
                in_b.append(matches_b[i, :])
                update += 1

        if update > Best_count:
            Best_count = update
            best_F = F
            inliers_a = in_a
            inliers_b = in_b

    inliers_a = np.array(inliers_a)
    inliers_b = np.array(inliers_b)

    return best_F, inliers_a, inliers_b
