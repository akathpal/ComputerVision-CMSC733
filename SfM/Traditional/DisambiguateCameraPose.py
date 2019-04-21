import numpy as np


def DisambiguateCameraPose( Cset, Rset, Xset):
    best = 0
    for i in range(4):

#         Cset[i] = np.reshape(Cset[i],(-1,-1))
        N = Xset[i].shape[0]
        n = 0
        for j in range(N):
            if ((np.dot(Rset[i][2, :],(Xset[i][j, :] - Cset[i])) > 0) and Xset[i][j, 2] >= 0):
                n = n+1
        if n > best:
            C = Cset[i]
            R = Rset[i]
            X = Xset[i]
            best = n

    return X, R, C
