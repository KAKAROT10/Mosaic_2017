import cv2
import numpy as np


for i in range (1,91):
    text = "lizard"+str(i)+".png"
    a = cv2.imread(text)
    grayc = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    blurredc = cv2.GaussianBlur(grayc, (5,5), 0)
    threshc = cv2.threshold(blurredc, 3, 255, cv2.THRESH_BINARY)[1]

    im2,contours,hierachy = cv2.findContours(threshc,1,2)
    cnt = contours[0]

    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(a,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imwrite(text,a[y:y+h,x:x+w])

