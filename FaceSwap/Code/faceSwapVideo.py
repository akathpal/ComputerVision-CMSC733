#! /usr/bin/env python

import sys
import numpy as np
import cv2
import dlib
import imutils
import random
import math
import imutils
from imutils import face_utils
from scipy.interpolate import interp2d

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    

    return points

def features(img):
    #initialize facial detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray,(300,300))
#     img = cv2.resize(img,(300,300))
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    rects = detector(gray,1)
    points = []
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(rects))
    for (i,rect) in enumerate(rects):
        
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)


        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        for (x,y) in shape:
            cv2.circle(img,(x,y),2,(0,0,255),-1)
            points.append((x,y))
            
#         points = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)
    return points



# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    # print(dst)

    return dst

def applyTransform(src,srcTri,dstTri,size):

    # print("src shape" + str(src.shape))
    # print("size" + str(size))

    t = dstTri
    s = srcTri

    r2 = cv2.boundingRect(np.float32([t]))

    xleft = r2[0]
    xright = r2[0] + r2[2]
    ytop = r2[1]
    ybottom = r2[1] + r2[3]
    # print(xleft,ytop,xright,ybottom)

    barytransform = np.linalg.inv([[t[0][0],t[1][0],t[2][0]],[t[0][1],t[1][1],t[2][1]],[1,1,1]])     

    grid = np.mgrid[xleft:xright, ytop:ybottom].reshape(2,-1)
    #grid 2x608
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    #grid 3x608
    barycoords = np.dot(barytransform, grid)
    # print(len(barycoords[0])) 608
    t = np.all(barycoords>=0, axis=0)
    # print(t.shape)
    dst_y = []
    dst_x = []
    for i in range(len(t)):
        if(t[i]):
            dst_y.append(i%r2[3])
            dst_x.append(i/r2[3])

            # print(i,i%r2[3],i%r2[2])

    # print(len(dst_x))
    # print(dst)

    # for i in xrange(len(barycoords[0])):
    #     print(np.sum(barycoords[:,i]))

    barycoords = barycoords[:,np.all(barycoords>=0, axis=0)]
  
    trans = np.matrix([[s[0][0],s[1][0],s[2][0]],[s[0][1],s[1][1],s[2][1]],[1,1,1]])
    pts = np.matmul(trans,barycoords)
    
    xA = pts[0,:]/pts[2,:]
    yA = pts[1,:]/pts[2,:]

    dst = np.zeros((size[1],size[0],3), np.uint8)

    i = 0
    for x,y in zip(xA.flat,yA.flat):
        xs = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)
        ys = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)

        b = src[:, :, 0]
        fb = interp2d(xs, ys, b, kind='cubic')

        g = src[:, :, 1]
        fg = interp2d(xs, ys, g, kind='cubic')

        r = src[:, :, 2]
        fr = interp2d(xs, ys, r, kind='cubic')

        blue = fb(x, y)[0]
        green = fg(x, y)[0]
        red = fr(x, y)[0]

        dst[dst_y[i],dst_x[i]] = (blue,green,red)
        i = i+1
        # print(blue,green, red)

    return dst

def U(r):
    u = (r**2)*np.log(r**2)

    if(math.isnan(u)):
        # print("NaN")
        u = 0
    # else:
    #     print(u)

    return u

# % Inputs:   ctr_pts     N x 2 matrix, each row representing corresponding
# %                       points position (x, y) in second image
# %           target_value    N x 1 vector representing corresponding point
# %                           position x or y in first image
# % Outputs:  a1          double, TPS parameter
# %           ax          double, TPS parameter
# %           ay          double, TPS parameter
# %           w           N x 1 vector, TPS parameters

def estimate_params(points2,points1_d):

    p = len(points2)

    K = np.zeros((p,p), np.float32)
    P = np.zeros((p,3), np.float32)

    
    for i in xrange(p):
        for j in xrange(p):
            a = points2[i,:]
            b = points2[j,:]
            K[i,j] = U(np.linalg.norm((a-b),ord=1))

    P = np.hstack((points2,np.ones((p,1))))
    # print("here")
    
    A = np.hstack((P.transpose(),np.zeros((3,3))))
    B = np.hstack((K,P))
    C = np.vstack((A,B))
    lamda = 0.0000001

    T = np.linalg.inv(C + lamda*np.identity(p+3))
    # print(T.shape)
    
    target = np.concatenate((points1_d,[0,0,0]))
    # print(target.shape)

    params = np.matmul(T,target)

    # print(params.shape)

    return params
    



def TPS(img1,img2,points1,points2):

    points1 = np.asarray(points1)
    points2 = np.asarray(points2)

    # print(points1[:,0])

    x_params = estimate_params(points2,points1[:,0])
    y_params = estimate_params(points2,points1[:,1])

    p = len(points1)

    K = np.zeros((p,p), np.float32)
    P = np.zeros((p,3), np.float32)

    
    for i in xrange(p):
        for j in xrange(p):
            a = points1[i,:]
            b = points1[j,:]
            K[i,j] = U(np.linalg.norm((a-b),ord=1))

    P = np.hstack((points1,np.ones((p,1))))
    # print("here")
    
    A = np.hstack((P.transpose(),np.zeros((3,3))))
    B = np.hstack((K,P))
    C = np.vstack((A,B))

    # print(x_params.shape)
    Ax = np.matmul(C,x_params)
    print(Ax)
    Ay = np.matmul(C,y_params)

    warped_img = []

    return warped_img

     


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri
        

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    # print(t1)
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    # print(r1)
    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    # print("Mask shape = "+str(mask.shape))

    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    # img2Rect = applyTransform(img1Rect, t1Rect, t2Rect, size)

    # print("img2rect shape =  "+str(img2Rect.shape))
    
    img2Rect = img2Rect * mask

    a = (1.0, 1.0, 1.0) - mask
    # print(a.shape)

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    

if __name__ == '__main__' :

    # Read images
    filename1 = '../Images/chetan.jpg'
    # filename2 = '../Images/yiannis.jpg'
    
    img1 = cv2.imread(filename1);
    # img2 = cv2.imread(filename2);
    # img1Warped = np.copy(img2);

    points1 = features(img1);
    # points2 = features(img2);

    cap = cv2.VideoCapture("../Images/1.mp4")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

    while(cap.isOpened()):

        ret,img2 = cap.read()
        if(ret==True):
            img2 = imutils.rotate(img2,90)
            img1Warped = np.copy(img2);

            # cv2.imshow("Frames",img2)
            # cv2.waitKey(10)
            # cv2.destroyWindow("Frames")
            points2 = features(img2)
            # warped = TPS(img1,img2,points1,points2)    
            
            # Find convex hull
            hull1 = []
            hull2 = []

            hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
            # print(len(hullIndex))
                  
            for i in xrange(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])
            
            
            # Find delanauy traingulation for convex hull points
            sizeImg2 = img2.shape    
            rect = (0, 0, sizeImg2[1], sizeImg2[0])
             
            dt = calculateDelaunayTriangles(rect, hull2)
            
            if len(dt) == 0:
                quit()
            
            # Apply affine transformation to Delaunay triangles
            for i in xrange(0, len(dt)):
                t1 = []
                t2 = []
                
                #get points for img1, img2 corresponding to the triangles
                for j in xrange(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])
                
                warpTriangle(img1, img1Warped, t1, t2)
                # if(i==10):
                #     break
            
            # cv2.imshow("Face Warped", img1Warped)
            # cv2.waitKey(100)

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
            
            cv2.imshow("Face Swapped", output)
            cv2.waitKey(1)
            # out_vid = 'video/cube/'+str(video)+'/'+str(count)+'.png'
            out.write(output)
            
            if cv2.waitKey(1) & 0xff==ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            break
        
        
