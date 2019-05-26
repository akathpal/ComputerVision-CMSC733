"""Summary
"""
import numpy as np
import sys

sys.dont_write_bytecode = True

def skew(x):
    """Summary

    Args:
        x (TYPE): Description

    Returns:
        TYPE: Description
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """Summary

    Args:
        K (TYPE): Description
        C1 (TYPE): Description
        R1 (TYPE): Description
        C2 (TYPE): Description
        R2 (TYPE): Description
        x1 (TYPE): Description
        x2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    I = np.identity(3)
    sz = x1.shape[0]
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    #     print(P2.shape)
    X1 = np.hstack((x1, np.ones((sz, 1))))
    X2 = np.hstack((x2, np.ones((sz, 1))))

    X = np.zeros((sz, 3))

    for i in range(sz):
        skew1 = skew(X1[i, :])
        skew2 = skew(X2[i, :])
        A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
        _, _, v = np.linalg.svd(A)
        x = v[-1] / v[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X[i, :] = x[0:3].T

    return X
