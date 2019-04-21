import numpy as np

def EstimateFundamentalMatrix(points_a, points_b):
    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num, 1))

    cu_a = np.sum(points_a[:, 0])/points_num
    cv_a = np.sum(points_a[:, 1])/points_num

    s = points_num/np.sum(((points_a[:, 0]-cu_a)**2 + (points_a[:, 1]-cv_a)**2)**(1/2))
    T_a =np.dot(np.array([[s, 0, 0], [0,s,0], [0,0,1]]), np.array([[1,0,-cu_a],[0,1,-cv_a],[0,0,1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a,B)

    points_a = np.reshape(points_a, (3,points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:, 0])/points_num
    cv_b = np.sum(points_b[:, 1])/points_num

    s = points_num/np.sum(((points_b[:,0]-cu_b)**2 + (points_b[:,1]-cv_b)**2)**(1/2))
    T_b =np.dot(np.array([[s,0,0], [0,s,0], [0,0,1]]), np.array([[1,0,-cu_b],[0,1,-cv_b],[0,0,1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b,B)

    points_b = np.reshape(points_b, (3,points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i,0]
        v_a = points_a[i,1]
        u_b = points_b[i,0]
        v_b = points_b[i,1]
        A.append([u_a*u_b, v_a*u_b, u_b, u_a*v_b, v_a*v_b, v_b, u_a, v_a,1])


    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    F = F/F[2, 2]

    return F
