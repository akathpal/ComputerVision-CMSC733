""" File to implement Non Linear PnP method
"""
import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as Rscipy


def reprojError(CQ, K, X, x):
    """Function to calculate reprojection error

    Args:
        K (TYPE): intrinsic matrix
        X (TYPE): 3D points
        x (TYPE): 2D points

    Returns:
        TYPE: Reprojection error
    """
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    C = CQ[0:3]
    R = CQ[3:7]
    C = C.reshape(-1, 1)
    r_temp = Rscipy.from_quat([R[0], R[1], R[2], R[3]])
    R = r_temp.as_dcm()

    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))

    # print("P",P.shape, X3D.shape)
    u_rprj = (np.dot(P[0, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    v_rprj = (np.dot(P[1, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    e1 = x[:, 0] - u_rprj
    e2 = x[:, 1] - v_rprj
    e = e1 + e2

    return sum(e)


def NonLinearPnP(X, x, K, C0, R0):

    q_temp = Rscipy.from_dcm(R0)
    Q0 = q_temp.as_quat()
    # reprojE = reprojError(C0, K, X, x)

    CQ = [C0[0], C0[1], C0[2], Q0[0], Q0[1], Q0[2], Q0[3]]
    assert len(CQ) == 7, "length of init in nonlinearpnp not matched"
    optimized_param = opt.least_squares(
        fun=reprojError, method="dogbox", x0=CQ, args=[K, X, x])
    Cnew = optimized_param.x[0:3]
    assert len(Cnew) == 3, "Translation Nonlinearpnp error"
    R = optimized_param.x[3:7]
    r_temp = Rscipy.from_quat([R[0], R[1], R[2], R[3]])
    Rnew = r_temp.as_dcm()

    return Cnew, Rnew
