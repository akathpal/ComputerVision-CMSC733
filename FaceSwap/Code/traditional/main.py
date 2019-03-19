from tps import thinPlateSpline
from triangulation import triangulation
from facial_landmarks import *
import numpy as np
import cv2

def blending(img1Warped,hull2,img2):
    # Calculate Mask
    hull8U = []
    for i in xrange(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    
    mask = np.zeros(img2.shape, dtype = img2.dtype)   
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255)) 
    r = cv2.boundingRect(np.float32([hull2]))     
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    return output

def traditional(img1,img2,points1,points2,method):
    img1Warped = np.copy(img2);

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
    # print(len(hullIndex))
          
    for i in xrange(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    if(method=="tps"):

        img1Warped = thinPlateSpline(img1,img1Warped,points1,points2,hull2)
        
        # cv2.imshow("Face Warped", img1warped)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows() 

    elif(method=="affine" or method=="tri"):

        img1Warped = triangulation(img1,img2,img1Warped,hull1,hull2,method)

    output = blending(img1Warped,hull2,img2)

    return output


