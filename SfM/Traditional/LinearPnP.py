""" File to implement Linear PnP method
"""
import numpy as np


def convertHomogeneouos(x):
    """Summary

    Args:
        x (array): 2D or 3D point

    Returns:
        TYPE: point appended with 1
    """
    m, n = x.shape
    if (n == 3 or n == 2):
        x_new = np.hstack((x, np.ones((m, 1))))
    else:
        x_new = x
    return x_new


def LinearPnP(X, x, K):
    """Summary

    Args:
        X (TYPE): 3D points
        x (TYPE): 2D points
        K (TYPE): intrinsic Matrix

    Returns:
        TYPE: C_set, R_set
    """
    N = X.shape[0]
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    x = np.transpose(np.dot(np.linalg.inv(K), x.T))
    A = []
    for i in range(N):
        xt = X[i, :].reshape((1, 4))
        z = np.zeros((1, 4))
        p = x[i, :]  #.reshape((1, 3))

        a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
        a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
        a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
        a = np.vstack((np.vstack((a1, a2)), a3))

        if (i == 0):
            A = a
        else:
            A = np.vstack((A, a))

    _, _, v = np.linalg.svd(A)
    P = v[-1].reshape((3, 4))
    R = P[:, 0:3]
    t = P[:, 3]
    u, _, v = np.linalg.svd(R)

    R = np.matmul(u, v)
    d = np.identity(3)
    d[2][2] = np.linalg.det(np.matmul(u, v))
    R = np.dot(np.dot(u, d), v)
    C = -np.dot(np.linalg.inv(R), t)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R
