import numpy as np


# def get_camera_param():
#     Cam = np.fromfile('Data/calibration_new.txt', sep=' ')
#     Cam = Cam.reshape(3, 3)
#     return(Cam)


def EssentialMatrixFromFundamentalMatrix(F, K):

    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)

    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E
