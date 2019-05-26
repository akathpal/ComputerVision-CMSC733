""" File to rectify camera poses
"""
import numpy as np
import sys

sys.dont_write_bytecode = True


def DisambiguateCameraPose(Cset, Rset, Xset):
    """ Function to implement camera pose correction

    Args:
        Cset (TYPE): Set of calculated camera poses
        Rset (TYPE): Set of calculated Rotation matrices
        Xset (TYPE): 3D points

    Returns:
        TYPE: Corrected X, R_set, C_set
    """
    best = 0
    for i in range(4):

        #         Cset[i] = np.reshape(Cset[i],(-1,-1))
        N = Xset[i].shape[0]
        n = 0
        for j in range(N):
            if ((np.dot(Rset[i][2, :], (Xset[i][j, :] - Cset[i])) > 0)
                    and Xset[i][j, 2] >= 0):
                n = n + 1
        if n > best:
            C = Cset[i]
            R = Rset[i]
            X = Xset[i]
            best = n

    return X, R, C
