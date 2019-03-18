import cv2
import dlib
from imutils import face_utils

def facial_landmarks(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./traditional/shape_predictor_68_face_landmarks.dat')
    rects = detector(gray,1)

    # cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    # preditor = dlib.shape_predictor('./mmod_human_face_detector.dat')
    # rects = cnn_face_detector(gray, 1)

    points = []
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    num_faces = len(rects)
    for (i,rect) in enumerate(rects):
        
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        (x,y,w,h) = face_utils.rect_to_bb(rect)

        # x = rect.rect.left()
        # y = rect.rect.top()
        # w = rect.rect.right() - x
        # h = rect.rect.bottom() - y
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        for (x,y) in shape:
            cv2.circle(img,(x,y),2,(0,0,255),-1)
            points.append((x,y))
            
    return num_faces,points