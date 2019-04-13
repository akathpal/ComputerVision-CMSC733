import numpy as np
import cv2
import glob


from misc.homographies import estimateHomographies
from misc.optimize import optimization
from misc.params import estimateIntrinsicParams
from misc.params import estimateExtrinsicParams

from misc.reproject import estimateReprojectionError,estimateReprojectionErrorDistortion
from misc.reproject import visualizePoints,rectify




if __name__ == '__main__':

	images = sorted(glob.glob('Calibration_Imgs/*.jpg'))

	#Pattern boxes in x and y direction excluding outer boundary
	Nx = 9
	Ny = 6

	homographies,imgpoints,objpoints = estimateHomographies(images,Nx,Ny)
	K_initial = estimateIntrinsicParams(homographies)
	print("Intial Intrinsic Camera Matrix is:", K_initial)

	extrinsics=[]
	err = []
	for im_points,H in zip(imgpoints,homographies):
		extrinsic = estimateExtrinsicParams(K_initial, H)
		err.append(estimateReprojectionError(K_initial,extrinsic,im_points,objpoints))
		extrinsics.append(extrinsic)
	# print("Extrinsics parameters for all images are:\n", extrinsics)
	
	print("Mean Reprojection error before optimization: ",np.mean(err))

	K_final,k1,k2 = optimization(K_initial,imgpoints,objpoints,homographies)

	optpoints=[]
	err = []
	for im_points,H in zip(imgpoints,homographies):
		extrinsic = estimateExtrinsicParams(K_final, H)
		error,points = estimateReprojectionErrorDistortion(K_final,extrinsic,im_points,objpoints,k1,k2)
		optpoints.append(points)
		err.append(error)
	
	visualize = True
	if visualize:
		visualizePoints(imgpoints, np.asarray(optpoints), images)

	rectified_images = rectify(imgpoints,optpoints,images)

	print("Mean Reprojection error after optimization:",np.mean(err))
	print("Final Calibration matrix is: ",K_final)
	print("\nDistortion coefficients after optimization are: ",k1, k2)

	




