import cv2
import numpy as np
image = cv2.imread('someshapes.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,1)
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for (i,c) in enumerate(contours):
    approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
#     cv2.drawContours(image,[c],0,(0,255,0),10)
    shape = ""
    if len(approx) == 3:
        shape = "Triangle"
        cv2.drawContours(image,[c],0,(0,255,0),-1)
        # Find contour center to place text at the center
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image, shape, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(c)
        if abs(w-h) <= 3:
            shape = "Square"
            cv2.drawContours(image,[c],0,(0, 125 ,255),-1)
            cv2.putText(image,shape,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            shape = "Rectangle"
            cv2.drawContours(image,[c],0,(0, 0, 255),-1)
            cv2.putText(image,shape,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    elif len(approx) == 10:
        shape = "Star"
        cv2.drawContours(image,[c],0,(0, 255, 0),-1)
        cv2.putText(image,shape,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    elif len(approx) >= 15:
        shape = "Circle"
        cv2.drawContours(image,[c],0,(0, 255, 255),-1)
        cv2.putText(image,shape,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.imshow('Identifying Shapes',image)
    cv2.waitKey(0)
    print(i,shape)
cv2.destroyAllWindows()