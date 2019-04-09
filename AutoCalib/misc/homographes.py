import numpy as np
import cv2


def findHomographies(images,Nx,Ny,display=False):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # objp = np.zeros((Ny*Nx,3), np.float32)
    # objp[:,:2] = np.mgrid[0:Nx,0:Ny].T.reshape(-1,2) #*21.5

    world_coord = []
    for i in range(1, board_size[0] - 1):
        for j in range(1, board_size[1] - 1):
            world_coord.append([21.5 * j, 21.5 * i])

    world_coord_x_y = np.array(world_coord)
    world_coord = np.c_[world_coord_x_y, np.zeros(len(world_coord))]
    print(world_coord)
    world_points = []
    world_points_x_y = []
    
    # objpoints = [] # 3d point in model plane.
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
            # objpoints.append(objp)
            corners=corners.reshape(-1,2)
            # assert corners.shape == objp[:, :-1].shape, "No. of Points not matched"
            # Refining the points
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            world_points.append(world_coord)
            world_points_x_y.append(world_coord_x_y)

            H,_ = cv2.findHomography(world_coord,corners2)
            homography.append(H)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)

            if(display):
                cv2.imwrite('Corners{}.png'.format(i),img)
                i=i+1
                cv2.imshow('img', img)
                cv2.waitKey(400)

    cv2.destroyAllWindows()

    return homography,imgpoints,world_points