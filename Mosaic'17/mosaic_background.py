import cv2
import numpy as np
import math
#import scipy.ndimage.
import matplotlib.pyplot as plt

ic=[]
refPt1=[]
colorsp=[]
ankur = 23
harshit=0
lower_skin=np.array([45,45,45])
upper_skin=np.array([100,100,100])

def on_mouse_click (event,x, y,flags,param):
#    key =cv2.waitKey(0) & 0xFF
    if event==cv2.EVENT_LBUTTONDOWN:
        ic.append((x,y))
        cropping1=True

    elif event==cv2.EVENT_LBUTTONUP:
        refPt1.append((x,y))
        cropping1=False

#        cv2.rectangle(image, ic[0], refPt1[0], (0, 255, 0), 2)
#        cv2.imshow("image", image)

        tr = ( (refPt1[0][0]+ic[0][0])/2,(refPt1[0][1]+ic[0][1])/2 )
        ic.append(tr)
        ar1 = final_out[int(ic[0][1])-2:int(ic[0][1])+2,int(ic[1][0])-2:int(ic[1][0])+2]
        ar2 = np.average(ar1, axis=0)
        ar3 = np.average(ar2, axis=0)
        colorsp.append(ar3)
        cv2.rectangle(final_out,ic[0],refPt1[0],(0,255,0),2)

        ic.pop()
        ic.pop()
        refPt1.pop()

vid1 = cv2.VideoCapture(0)
ret, background=vid1.read()
vid1.release()

cap =  cv2.VideoCapture(0)
while(1):


    ret, frame = cap.read()
    frame1=cv2.medianBlur(frame,51)
    fg=frame-background
    #cv2.imshow('fg',fg)

    fgray=cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)

    #gaus=cv2.adaptiveThreshold(fg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
    ret,thresh=cv2.threshold(fgray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #fg=cv2.cvtColor(fg,cv2.COLOR_GRAY2)
    cv2.imshow('ttrsh',thresh)

    kernel=np.ones((5,5),np.uint8)
    #thresh=cv2.dilate(thresh,kernel,iterations=1)
    thresh=cv2.erode(thresh,kernel,iterations=3)
    thresh=cv2.GaussianBlur(thresh,(15,15),0)
    cv2.imshow('thresh',thresh)

    mask=np.zeros(thresh.shape,np.uint8)
    image,contours,heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # cnt = max(contours, key = cv2.contourArea)
    # #print "000000",cv2.contourArea(cnt)
    # cv2.drawContours(mask, cnt, 1, 255, 100)
    largest_area=sorted(contours,key=cv2.contourArea)
    if len(largest_area)>2 :
        cv2.drawContours(mask, [largest_area[-1]] , 0, 255 , -1)

    print "----"

    mask=cv2.erode(mask,kernel,iterations=2)
    #cv2.imshow('mask',mask)
    res1=cv2.bitwise_and(frame,frame,mask=mask)
    #cv2.imshow('out1',res1)

    # fhand=cv2.GaussianBlur(thresh,(25,25),0)
    # cv2.imshow('threshold',fhand)
    #
    #
    # #cv2.imshow('foreground',fg)
    #res2=cv2.Canny(res1,35,40,100)
    #res2=cv2.GaussianBlur(res2,(5,5),0)
    #cv2.imshow('res2',res2)
    #res2=cv2.dilate(res2,kernel,iterations=1)
    #cv2.imshow('res22',res2)
    #plt.imshow(res1,cmap='gray',interpolation='bicubic')
    #plt.show()

    cv2.imshow('out', res1)


    final_out=cv2.inRange(res1,lower_skin,upper_skin)
    # cv2.imshow('out1', res2)
    # #mask_final=cv2.dilate(res2,kernel,iterations=1)
    # final_out=cv2.bitwise_and(frame,frame,mask=res2)
    # #final_out=cv2.GaussianBlur(final_out,(5,5),0)
    # final_out=cv2.morphologyEx(final_out,cv2.MORPH_OPEN,kernel,iterations=1)
    # final_out=cv2.medianBlur(final_out,5)
    # cv2.imshow('output',final_out)
    #
    # #cv2.imshow('out2',final_out)
    # #final_out=cv2.morphologyEx(final_out,cv2.MORPH_CLOSE,kernel)
    #
    # # images, contours, hierarchy = cv2.findContours(mask_final,2,1)
    # # if len(contours)>0:
    # #     cnt=max(contours,key=cv2.contourArea)
    # # emp=[]
    # # if len(contours)>0:
    # #     cnt=max(contours,key=cv2.contourArea)
    # #     #print cv2.contourArea(cnt)
    # #     hull = cv2.convexHull(cnt,returnPoints=False)
    # #     #if not hull is None:
    # #
    # #     #cv2.drawContours(frame,hull,0,(0,255,0),3)
    # #     if not hull is None:
    # #         defects = cv2.convexityDefects(cnt,hull)
    # #     #emp=[]
    # #     if not defects is None:
    # #         for i in range(defects.shape[0]):
    # #             s,e,f,d = defects[i,0]
    # #             start = tuple(cnt[s][0])
    # #             end = tuple(cnt[e][0])
    # #             far = tuple(cnt[f][0])
    # #             #emp.append(far)
    #
    #             #cv2.line(final_out,start,end,[0,255,0],2)
    #             #print far
    #
    #             #emp.append(far)
    #             #cv2.circle(final_out,far,5,[0,0,255],-1)
    # #     #
    # #     #         #for j in range(1,len(emp)-1):
    # #     #         #    cv2.line(fg,emp[j],emp[j+1],[0,255,0],2)
    # #     # print "0-0-0-0-0-0-0"
    # #     # for j in range(1,len(emp)-2):
    # #     #     a0=math.atan2(emp[j][1]-emp[j-1][1],emp[j][0]-emp[j-1][0])
    # #     #     a1=math.atan2(emp[j+1][1]-emp[j][1],emp[j+1][0]-emp[j][0])
    # #     #     angle=(a1-a0)*180/3.14
    # #     #     if angle>20:
    # #     #         cv2.circle(res1,emp[j+1],5,[0,0,255],-1)
    # #     #     print angle
    # #     # print "0-0-0-0-0-0-0"
    #
    final_out=cv2.bitwise_and(frame,frame,mask=final_out)
    final_out=cv2.morphologyEx(res1,cv2.MORPH_OPEN,kernel,iterations=3)
    # #final_out1=cv2.medianBlur(final_out,15)
    #
    # cv2.imshow('output',final_out)

    # res2=cv2.cvtColor(res1,cv2.COLOR_RGB2YCrCb)
    #plt.imshow(res2,cmap='gray',interpolation='bicubic')
    #plt.show()

    #res2=cv2.inRange(final_out,lower_skin,upper_skin)
    #final_out=cv2.cvtColor(final_out,cv2.COLOR_YCrCb2BGR)
    #final_out=cv2.cvtColor(final_out,cv2.COLOR_BGR2GRAY)
    #ret,final_mask=cv2.threshold(final_out,100,0,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #final_out=cv2.bitwise_and(final_out,final_out,mask=final_mask)


    #res2=cv2.Canny(res1,156,167,40)
    cv2.imshow("final_out",final_out)
    k=cv2.waitKey(1)

    # if(k==ord('q') and harshit == 0):
    #     harshit = harshit+1
    #     cv2.setMouseCallback("final_out", on_mouse_click)
#     else:
#     if k == ord('q') and ankur==23:
# #        if ankur==2:
#             cv2.imwrite("scissor"+str(ankur)+".png",final_out)
#             ankur=ankur+1
#
#     elif k == ord('q') and ankur>23:
#         cap.release()
#         cv2.destroyAllWindows()
#         break
    elif k==ord('q') and harshit == 1:
           print colorsp
           lower_skin = np.array([int(colorsp[0][0]) - 20, int(colorsp[0][1]) - 20, int(colorsp[0][2]) - 20])
           upper_skin = np.array([int(colorsp[0][0]) + 20, int(colorsp[0][1]) + 20, int(colorsp[0][2]) + 20])
           harshit=harshit+1
    elif k==ord('q'):
           cv2.imwrite("don.png",final_out)
           cap.release()
           cv2.destroyAllWindows()
           break