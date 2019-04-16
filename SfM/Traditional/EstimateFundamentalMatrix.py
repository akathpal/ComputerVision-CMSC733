import numpy as np 

def EstimateFundamentalMatrix(x1,x2):

	sz = x1.shape[0]
	x1_x = x1[:,0]
	x2_x = x2[:,0]
	x1_y = x1[:,1]
	x2_y = x2[:,1]

	# Scaling for image points for first image
	cent1_x = np.mean(x1[:,0])
	cent1_y = np.mean(x1[:,1])
	x1_x = x1_x - cent1_x * np.ones([sz,1])
	x1_y = x1_y - cent1_y * np.ones([sz,1])
	avg_dist = np.sqrt(np.sum(np.power(x1_x,2)  + np.power(x1_y,2))) / sz
	scaling_1 = sqrt(2) / avg_dist
	x1[:,0] = scaling_1 * x1_x
	x1[:,1] = scaling_1 * x1_y


	Scaled_1 = np.array([scaling_1,         0,(-scaling_1*cent1_x)],
	                    [0, scaling_1,(-scaling_1*cent1_y)],
	                    [0,         0,                    1])

	# Scaling for image points for first image
	cent2_x = np.mean(x2_x)
	cent2_y = np.mean(x2_y)
	x2_x = x2_x - cent2_x * np.ones([sz,1])
	x2_y = x2_y - cent2_y * np.ones([sz,1])
	avg_dist = np.sqrt(np.sum(np.power(x2_x,2)  + np.power(x2_y,2))) / sz
	scaling_2 = sqrt(2) / avg_dist
	x2[:,0] = scaling_2 * x2_x
	x2[:,1] = scaling_2 * x2_y
	Scaled_2 = np.array([scaling_2 0 -scaling_2*cent2_x],
	       [0 scaling_2 -scaling_2*cent2_y],
	       [0 0 1])


	W = [x1[:,0]*x2[:,0] x1[:,0]*x2[:,1] x1[:,1] x1[:,1]*x2[:,0] x1[:,1]*x2[:,1] x1[:,1] x2[:,0] x2[:,1] np.ones([sz,1])];
	_,_,v = np.linalg.svd(W);
	F = v[:,end];
	F = np.reshape(F,(3,3))
	F_norm = F / np.linalg.norm(F);
	uf,sf,vf = np.linalg.svd(F_norm);

	F_final = uf*np.diag([sf(1) sf(5) 0])*vf.T

	# Denormalizing
	F = Scaled_2.T * F_final * Scaled_1

	return F