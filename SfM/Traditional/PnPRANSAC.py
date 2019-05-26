"""Summary
"""
import numpy as np
import LinearPnP as LPnP
import random
from tqdm import tqdm


def proj3Dto2D(x3D, K, C, R):
    """Summary

    Args:
        x3D (TYPE): Description
        K (TYPE): Description
        C (TYPE): Description
        R (TYPE): Description

    Returns:
        TYPE: Description
    """
    C = C.reshape(-1, 1)
    x3D = x3D.reshape(-1, 1)
    # print("K", K.shape, R.shape, C.shape, x3D.shape)
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    X3D = np.vstack((x3D, 1))

    # print("P",P.shape, X3D.shape)
    u_rprj = (np.dot(P[0, :], X3D)).T / (np.dot(P[2, :], X3D)).T
    v_rprj = (np.dot(P[1, :], X3D)).T / (np.dot(P[2, :], X3D)).T
    X2D = np.hstack((u_rprj, v_rprj))
    return X2D


def PnPRANSAC(X, x, K):
    """Summary

    Args:
        X (TYPE): Description
        x (TYPE): Description
        K (TYPE): Description

    Returns:
        TYPE: Description
    """
    cnt = 0
    M = x.shape[0]
    threshold = 5  #6
    x_ = LPnP.convertHomogeneouos(x)

    Cnew = np.zeros((3, 1))
    Rnew = np.identity(3)

    for trails in tqdm(range(500)):
        # random.randrange(0, len(corr_list))
        random_idx = random.sample(range(M), 6)
        C, R = LPnP.LinearPnP(X[random_idx][:], x[random_idx][:], K)
        S = []
        for j in range(M):
            reprojection = proj3Dto2D(x_[j][:], K, C, R)
            e = np.sqrt(
                np.square((x_[j, 0]) - reprojection[0]) +
                np.square((x_[j, 1] - reprojection[1])))
            if e < threshold:
                S.append(j)
        countS = len(S)
        if (cnt < countS):
            cnt = countS
            Rnew = R
            Cnew = C

        if (countS == M):
            break
    # print("Inliers = " + str(cnt) + "/" + str(M))
    return Cnew, Rnew
