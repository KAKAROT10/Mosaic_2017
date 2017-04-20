import cv2
import numpy as np
import math
#import scipy.ndimage.
import matplotlib.pyplot as plt



# #-----ML
# import pandas
# import numpy as np
# from sklearn import cross_validation
# from sklearn.cross_validation import KFold
# from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
#
# featureX = pandas.read_csv("featureX.csv")
# y = pandas.read_csv("featureY.csv")
#
# predictors=[]
# for i in range(1,2501):
#     predictors.append(str(i))
#
# #alg = RandomForestClassifier(random_state=1, n_estimators=25, min_samples_split=8, min_samples_leaf=4)
# alg = LogisticRegression(random_state=1)
# predictions=[]
# kf = KFold(featureX.shape[0], n_folds=20, random_state=1)
#
# for train, test in kf:
#     #print "gain"
#     # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
#     train_predictors = (featureX[predictors].iloc[train,:])
#     # The target we're using to train the algorithm.
#     train_target = y["result"].iloc[train]
#     # Training the algorithm using the predictors and target.
#     alg.fit(train_predictors, train_target)
#     # We can now make predictions on the test fold
#     test_predictions = alg.predict(featureX[predictors].iloc[test,:])
#     predictions.append(test_predictions)
#
# scores = cross_validation.cross_val_score(alg, featureX[predictors], y["result"], cv=kf)
# print(scores.mean())

#-----ML





ic=[]
refPt1=[]
colorsp=[]
ankur = 1
harshit=61
lower_skin=np.array([40,40,30])
upper_skin=np.array([180,200,200])






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

    ret,frame = cap.read()
    #frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #plt.imshow(frame1,cmap='gray',interpolation='bicubic')
    #plt.show()
    #fr=cv2.inRange(frame1,lower_skin,upper_skin)
    #cv2.imshow('fr',fr)

    #images, contours, hierarchy = cv2.findContours(fr,2,1)
    # if len(contours)>0:
    #      cnt=max(contours,key=cv2.contourArea)
    # emp=[]
    # if len(contours)>0:
    #     cnt=max(contours,key=cv2.contourArea)
    #     print cv2.contourArea(cnt)
    #     hull = cv2.convexHull(cnt,returnPoints=False)
    #     #if not hull is None:
    #         #cv2.drawContours(frame,hull,0,(0,255,0),3)
    #     if not hull is None:
    #         defects = cv2.convexityDefects(cnt,hull)
    #         #emp=[]
    #     if not defects is None:
    #          for i in range(defects.shape[0]):
    #              s,e,f,d = defects[i,0]
    #              start = tuple(cnt[s][0])
    #              end = tuple(cnt[e][0])
    #              far = tuple(cnt[f][0])
    #              #emp.append(far)
    #
    #              cv2.line(frame,start,end,[0,255,0],2)
    #          #print far
    #
    #              #emp.append(far)
    #              cv2.circle(frame,far,5,[0,0,255],-1)
    #       #
    #      #         #for j in range(1,len(emp)-1):
    #           #    cv2.line(fg,emp[j],emp[j+1],[0,255,0],2)
    #      # print "0-0-0-0-0-0-0"
    #
    #     cv2.imshow('frame',frame)






    #frame1=cv2.medianBlur(frame,)

    fg = frame-background
    cv2.imshow('fg',fg)

    fgray=cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)

    #gaus=cv2.adaptiveThreshold(fg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
    ret,thresh=cv2.threshold(fgray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #fg=cv2.cvtColor(fg,cv2.COLOR_GRAY2)
    cv2.imshow('ttrsh',thresh)

    kernel=np.ones((5,5),np.uint8)
    #thresh=cv2.dilate(thresh,kernel,iterations=1)
    #thresh=cv2.morphologyEx(thresh,kernel,cv2.MORPH_CLOSE,iterations=1)
    thresh=cv2.GaussianBlur(thresh,(5,5),1)
    thresh=cv2.erode(thresh,kernel,iterations=1)
    thresh=cv2.medianBlur(thresh,15)
    thresh=cv2.erode(thresh,kernel,iterations=1)

    cv2.imshow('thresh',thresh)


    mask=np.zeros(thresh.shape,np.uint8)
    image,contours,heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # cnt = max(contours, key = cv2.contourArea)
    # #print "000000",cv2.contourArea(cnt)
    # cv2.drawContours(mask, cnt, 1, 255, 100)
    largest_area=sorted(contours,key=cv2.contourArea)
    if len(largest_area)>2 :
        cv2.drawContours(mask, [largest_area[-1]] , 0, 255 , -1)

    cv2.imshow('masssk',mask)

    #mask=cv2.erode(mask,kernel,iterations=1)
    #mask=cv2.medianBlur(mask,5)
    #mask=cv2.GaussianBlur(mask,(5,5),0)
    #k1=np.ones((3,3),np.uint8)
    #mask=cv2.erode(mask,k1,iterations=1)
    #cv2.imshow('masssk11',mask)
    #mask=cv2.medianBlur(mask,5)

    # mask_1=cv2.erode(mask,kernel,iterations=1)
    # mask_1=cv2.medianBlur(mask_1,5)
    # foregrnd=cv2.bitwise_and(frame,frame,mask=mask_1)
    # #plt.imshow(foregrnd,cmap='gray',interpolation='bicubic')
    # #plt.show()
    # fgd=cv2.cvtColor(foregrnd,cv2.COLOR_BGR2GRAY)
    # ret,hand=cv2.threshold(fgd,150,255,cv2.THRESH_OTSU)
    # #hand=cv2.inRange(foregrnd,lower_skin,upper_skin)
    #
    # hand=cv2.medianBlur(hand,5)
    # hand=cv2.erode(hand,kernel,iterations=3)
    #cv2.imshow('hand',hand)

    #cv2.imshow('mask',mask)
    #res1=cv2.bitwise_and(frame,frame,mask=mask)

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
    #cv2.imshow('out', res1)


    #final_out=cv2.inRange(res1,lower_skin,upper_skin)
    #cv2.imshow('out11', final_out)




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


    #final_out=cv2.bitwise_and(frame,frame,mask=final_out)
    #final_out=cv2.morphologyEx(res1,cv2.MORPH_OPEN,kernel,iterations=3)



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






    cv2.imshow("final_out",mask)
    img,contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        cnt=max(contours,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(cnt)
        mask=mask[y:y+h,x:x+w]
    cv2.imshow('ans',mask)




    # mask_out = cv2.resize(mask, (50, 50))
    # #mask_out=cv2.inRange(mask_out,lower,upper)
    # mask_out=np.array(mask_out)
    # mask_out=mask_out.ravel()
    # mask_out=mask_out/255
    #
    # final=alg.predict(mask_out)
    # if final[0]==0:
    #     print ""
    # if final[0]==1:
    #     print ""
    # if final[0]==2:
    #     print ""
    # if final[0]==3:
    #     print ""
    # if final[0]==4:
    #     print ""






    k=cv2.waitKey(1)
    if k==ord('q'):# and harshit <91:
          #cv2.imwrite('rock'+str(harshit)+'.jpg',mask)
          #harshit = harshit+1
    #elif k == ord('q') and harshit>=91:
          cap.release()
          cv2.destroyAllWindows()



    # elif k==ord('q') and harshit == 1:
    #        print colorsp
    #        lower_skin = np.array([int(colorsp[0][0]) - 20, int(colorsp[0][1]) - 20, int(colorsp[0][2]) - 20])
    #        upper_skin = np.array([int(colorsp[0][0]) + 20, int(colorsp[0][1]) + 20, int(colorsp[0][2]) + 20])
    #        harshit=harshit+1
    # elif k==ord('q'):
    #        cv2.imwrite("don.png",final_out)
    #        cap.release()
    #        cv2.destroyAllWindows()
    #        break