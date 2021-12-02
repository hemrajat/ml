import cv2
import dlib
from google.colab.patches import cv2_imshow
import numpy 
PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im,1)
    faces = []
    for i in range(len(rects)):
        faces.append(numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()]))
    return faces

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for landmark in landmarks:
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, (0, 255, 255),-1)
    return im

image = cv2.imread('swami_vivekanand.jpg')
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
cv2.imshow('Face landmarks',image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows() 