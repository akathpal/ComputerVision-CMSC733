""" File to implement function to calculate Essential Matrix
"""
import numpy as np
import sys

sys.dont_write_bytecode = True


def EssentialMatrixFromFundamentalMatrix(F, K):
    """Calculate essential matrix from Fundamental Matrix

    Args:
        F (array): Fundamental Matrix
        K (array): Camera Intrinsic Matrix

    Returns:
        array: Essential Matrix
    """
    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)

    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E
