import numpy as np
import cv2


def estimateHomographies(images,Nx,Ny,display=False):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    
    x, y = np.meshgrid(range(9), range(6))
    objpoints = np.hstack((x.reshape(54, 1), y.reshape(
        54, 1))).astype(np.float32)
    objpoints = objpoints*21.5
    objpoints = np.asarray(objpoints)

   
    imgpoints = [] # 2d points in image plane.
    homography =[]
    
    # print(objp[:, :-1])
    i=0
    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Finding ChessBoard Corners
        ret, corners = cv2.findChessboardCorners(gray, (Nx,Ny),None)

        if ret == True:
            
            corners=corners.reshape(-1,2)
            # assert corners.shape == objp[:, :-1].shape, "No. of Points not matched"
            # Refining the points
            corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            H,_ = cv2.findHomography(objpoints[:30],corners[:30])
            homography.append(H)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

            if(display):
                cv2.imwrite('Output/Corners{}.png'.format(i),img)
                i=i+1
                cv2.imshow('img', img)
                cv2.waitKey(400)

    cv2.destroyAllWindows()

    return homography,imgpoints,objpoints

