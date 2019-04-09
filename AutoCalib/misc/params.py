import numpy as np
import cv2


def v(p, q, H):
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])


def estimateIntrinsicParams(homographies):

    V = []

    for H in homographies:
    	V.append(v(0, 1, H))
    	V.append(v(0, 0, H) - v(1, 1, H))

    V = np.array(V)

    u, s, vh = np.linalg.svd(V)

    b = vh[np.argmin(s)]
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11

    alpha = np.sqrt(lamda/B11)
    beta = np.sqrt(lamda*B11 /(B11*B22 - B12**2))
    gamma = -1*B12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta -B13*(alpha**2)/lamda

    K = np.array([[alpha, gamma, u0],[0,     beta,  v0],[0,     0,      1]])

    return K

def estimateExtrinsicParams(K, H):

    K_inv = np.linalg.inv(K)

    #Rotation vectors

    
    r1 = np.dot(K_inv, H[:,0])
    lamda = np.linalg.norm(r1, ord=2)
    r1 = r1/lamda
    
    r2 = np.dot(K_inv, H[:,1])
    r2 = r2/lamda

    r3 = np.cross(r1, r2)

    #Translation vectors
    
    t = np.dot(K_inv, H[:,2])/lamda

    R = np.asarray([r1, r2, r3])
    R = R.T

    extrinsic = np.zeros((3, 4))
    extrinsic[:, :-1] = R
    extrinsic[:, -1] = t
    
    return extrinsic