import cv2
image = cv2.imread('/Users/hemraj/Downloads/lion.jpeg')
canny = cv2.Canny(image,150,200)
cv2.imshow('Lion Sketch',canny)
cv2.imshow('Lion ',image)
cv2.waitKey(1000)
#if you want to see the live sketch of yourself from the webcame then uncomment the code from line 8 to 16.
# live_vid = cv2.VideoCapture(0)
# while(True):
#     ret,frame = live_vid.read()
#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     canny = cv2.Canny(frame,150,200)
#     cv2.imshow('Live Skatch',canny)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# live_vid.release()
cv2.destroyAllWindows()
