import numpy as np
import math
from scipy import optimize as opt
from params import estimateExtrinsicParams

def minimizeFunction(init, imgpoints, objpoints, homographies):
    K = np.zeros(shape=(3, 3))
    
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1], K[2,2] = init[0], init[1], init[2], init[3], init[4], 1
    k1, k2 = init[5], init[6]
    u0, v0 = init[2], init[3]

    error = []
    i = 0
    for im_pts, H in zip(imgpoints, homographies):
        extrinsic = estimateExtrinsicParams(K, H)

        for pt, objpt in zip(im_pts, objpoints):
            model = np.array([[objpt[0]], [objpt[1]], [0], [1]])
            
            proj_point = np.dot(extrinsic, model)
            proj_point = proj_point/proj_point[2]
            x, y = proj_point[0], proj_point[1]

            U = np.dot(K, proj_point)
            U = U/U[2]
            u, v = U[0], U[1]

            t = x**2 + y**2
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))
            
            error.append(pt[0]-u_cap)
            error.append(pt[1]-v_cap)
          

    return np.float64(error).flatten()

def optimization(K,imgpoints,objpoints,homographies):

    alpha = K[0, 0]
    beta= K[1, 1]
    u0 = K[0, 2]
    v0 = K[1, 2]
    gamma = K[0, 1]
    
    initialization = [alpha, beta, u0, v0, gamma, 0, 0]
    optimized_params = opt.least_squares(fun=minimizeFunction, x0=initialization,method="lm", args=[imgpoints, objpoints, homographies])

    [alpha, beta, u0, v0, gamma, k1, k2] = optimized_params.x
   
    K = np.array([[alpha, gamma, u0],[0,     beta,  v0],[0,     0,      1]])

    return K,k1,k2
