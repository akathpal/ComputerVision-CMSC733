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
import argparse


def facial_landmarks(img):
    
    #initialize facial detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    rects = detector(gray,1)
    points = []
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    num_faces = len(rects)
    for (i,rect) in enumerate(rects):
        
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)


        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        for (x,y) in shape:
            cv2.circle(img,(x,y),2,(0,0,255),-1)
            points.append((x,y))
            
    return num_faces,points



# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def applyTransform(src,srcTri,dstTri,size):

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
    #grid 2xN
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    #grid 3xN
    barycoords = np.dot(barytransform, grid)
    
    t = np.all(barycoords>=0, axis=0)
    dst_y = []
    dst_x = []
    for i in range(len(t)):
        if(t[i]):
            dst_y.append(i%r2[3])
            dst_x.append(i/r2[3])

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
def warpTriangle(img1, img2, t1, t2,method) :

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

    if method == "affine":
        img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    else:
        img2Rect = applyTransform(img1Rect, t1Rect, t2Rect, size)

    # print("img2rect shape =  "+str(img2Rect.shape))
    
    img2Rect = img2Rect * mask

    a = (1.0, 1.0, 1.0) - mask
    # print(a.shape)

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect


def U(r):
    u = (r**2)*np.log(r**2)
    if(math.isnan(u)):
        u = 0
    return u

def estimateParams(points2,points1_d):

    p = len(points2)

    K = np.zeros((p,p), np.float32)
    P = np.zeros((p,3), np.float32)
 
    for i in xrange(p):
        for j in xrange(p):
            a = points2[i,:]
            b = points2[j,:]
            K[i,j] = U(np.linalg.norm((a-b)))

    P = np.hstack((points2,np.ones((p,1))))
    
    A = np.hstack((P.transpose(),np.zeros((3,3))))
    B = np.hstack((K,P))
    C = np.vstack((B,A))
    lamda = 0.0000001

    T = np.linalg.inv(C + lamda*np.identity(p+3))  
    target = np.concatenate((points1_d,[0,0,0])) 
    params = np.matmul(T,target)

    return params

def thinPlateSpline(img1,img2,points1,points2,hull2):

    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    p = len(points1)

    print(p)

    r = cv2.boundingRect(np.float32([points2]))
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)

    points2_t = []

    for i in xrange(len(hull2)):
        points2_t.append(((hull2[i][0]-r[0]),(hull2[i][1]-r[1])))

    cv2.fillConvexPoly(mask, np.int32(points2_t), (1.0, 1.0, 1.0), 16, 0);
    
    # cv2.imshow("Mask",mask)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    
    x_params = estimateParams(points2,points1[:,0])
    y_params = estimateParams(points2,points1[:,1])

    a1_x = x_params[p+2]
    ay_x = x_params[p+1]
    ax_x = x_params[p]

    a1_y = y_params[p+2]
    ay_y = y_params[p+1]
    ax_y = y_params[p]

    warped_img = np.copy(mask)

    for i in xrange(warped_img.shape[1]):
        for j in xrange(warped_img.shape[0]):
            t = 0
            l = 0
            n = i+r[0]
            m = j+ r[1]
            b = [n,m]
            for k in xrange(p):
                a = points2[k,:]
                t = t+x_params[k]*U(np.linalg.norm((a-b)))
                l = l+y_params[k]*U(np.linalg.norm((a-b)))

            x = a1_x + ax_x*n + ay_x*m + t
            y = a1_y + ax_y*n + ay_y*m + l

            x = int(x)
            y = int(y)
            x = min(max(x, 0), img1.shape[1] - 1)
            y = min(max(y, 0), img1.shape[0] - 1)

            warped_img[j,i] = img1[y,x,:]

    # cv2.imshow("warped_img",warped_img)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    warped_img = warped_img * mask

    img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] + warped_img

    return img2
    
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

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Face Swapping')

    parser.add_argument('--input_path', default="../TestSet/", type=str,help='path to the input')
    parser.add_argument('--face', default='Rambo', type=str,help='path to face')
    parser.add_argument('--video', default='Test1', type=str,help='path to the input video')
    parser.add_argument('--method', default='affine', type=str,help='affine, tri, tps, prnet')


    Args = parser.parse_args()
    video_path = Args.input_path+Args.video+'.mp4'
    face_path = Args.input_path+Args.face+'.jpg'
    video = Args.video
    method = Args.method

    
    img1 = cv2.imread(face_path);
    # img1 = imutils.resize(img1,width = 320)
    faces_num,points1 = facial_landmarks(img1)
    if(faces_num!=1):
        print("More than 1 or zero face detected...Exiting")
        exit()
    

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("No. of frames = "+str(length))

    
    ret,img2 = cap.read()
    height = img2.shape[0]
    width = img2.shape[1]

    # Defining Video Writer Object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('{}_Output_{}.avi'.format(method,video),fourcc, 20, (width,height))
    count = 0
    while(cap.isOpened()):

        ret,img2 = cap.read()
        if(ret==True):

            # img2 = imutils.rotate(img2,90)
            img1Warped = np.copy(img2);

            # cv2.imshow("Frames",img2)
            # cv2.waitKey(10)
            # cv2.destroyWindow("Frames")
            faces_num,points2 = facial_landmarks(img2)
            if(faces_num==0):
                # print("Face Not Detected")
                count+=1
                continue
            # warped = TPS(img1,img2,points1,points2)    
            
            # Find convex hull
            hull1 = []
            hull2 = []

            hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
            # print(len(hullIndex))
                  
            for i in xrange(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])

            if(method=="tps"):

                img1warped = thinPlateSpline(img1,img1Warped,points1,points2,hull2)
                
                # cv2.imshow("Face Warped", img1warped)
                # cv2.waitKey(2000)
                # cv2.destroyAllWindows() 

            elif(method=="affine" or method=="tri"):


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
                    
                    warpTriangle(img1, img1Warped, t1, t2,method)

            output = blending(img1Warped,hull2,img2)
            
            # cv2.imshow("Face Swapped", output)
            # cv2.waitKey(100)
            out.write(output)
            
            if cv2.waitKey(1) & 0xff==ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            break

    print(count)
        
        
