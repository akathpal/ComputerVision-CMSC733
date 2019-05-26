"""This file implements the RANSAC algorithm on given matches
"""
import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
import sys

sys.dont_write_bytecode = True


def GetInliersRANSAC(matches_a, matches_b, indices):
    """Function to implement RANSAC

    Args:
        matches_a (np.array(int32)): Matches of image1
        matches_b (np.array(int32)): Matches of image2
        indices (array): indices of points to consider

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
        ind = []
        update = 0
        for i in range(matches_num):
            matches_aa = np.append(matches_a[i, :], 1)
            matches_bb = np.append(matches_b[i, :], 1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < 0.005:
                in_a.append(matches_a[i, :])
                in_b.append(matches_b[i, :])
                ind.append(indices[i])
                update += 1

        if update > Best_count:
            Best_count = update
            best_F = F
            inliers_a = in_a
            inliers_b = in_b
            inlier_index = ind

    inliers_a = np.array(inliers_a)
    inliers_b = np.array(inliers_b)
    inlier_index = np.array(inlier_index)

    return best_F, inliers_a, inliers_b, inlier_index
