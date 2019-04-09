import numpy as np
import cv2

def estimateReprojectionError(K,extrinsic,imgpoints,objpoints):

	# Compute projection matrix
	P = np.dot(K,extrinsic)
	err = []
	
	for im_pts,obj_pts in zip(imgpoints,objpoints):
		
		model_pts = np.array([[obj_pts[0]], [obj_pts[1]], [0], [1]])
		real_pts = np.array([[im_pts[0]], [im_pts[1]], [1]])
		#         print(model_pts)
		proj_points = np.dot(P,model_pts)
		#         print(proj_points)
		proj_points = proj_points/proj_points[2]

		err.append(np.linalg.norm(real_pts-proj_points, ord=2))
	
	# print(err)
	return np.mean(err)


def estimateReprojectionErrorDistortion(K,extrinsic,imgpoints, objpoints, k1, k2):
    
	err = []
	reproject_points = []

	u0, v0 = K[0, 2], K[1, 2]
	

	for impt, objpt in zip(imgpoints, objpoints):
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

	    reproject_points.append([u_cap, v_cap])

	    err.append(np.sqrt((impt[0]-u_cap)**2 + (impt[1]-v_cap)**2))

	return np.mean(err), reproject_points


def visualizePoints(imgpoints, optpoints, images):

    for i, (image, imgpoints, optpoints) in enumerate(zip(images, imgpoints, optpoints)):
        img = cv2.imread(image)

        for im_pt, opt_pt in zip(imgpoints, optpoints):
            [x, y] = np.int64(im_pt)
            [x_correct, y_correct] = np.int64(opt_pt)
            cv2.rectangle(img, (x-5, y-5),(x+5,y+5), (0, 0, 255),thickness=cv2.FILLED)
            cv2.rectangle(img, (x_correct-5, y_correct-5), (x_correct+5, y_correct+5), (0, 255, 0), thickness=cv2.FILLED)

        cv2.imwrite("Output/reproj_{}.jpg".format(i), img)

