import cv2
import numpy as np
import math
#import scipy.ndimage.
import matplotlib.pyplot as plt


def determiner(x):
    if x==0:
        return "rock"
    elif x==1:
        return "paper"
    elif x==2:
        return "scissor"
    elif x==3:
        return "spock"
    elif x==4:
        return "lizard"

def win_or_lose(x1,x2):
    if x1==0 and x2==1:
        return 1
    elif x1==0 and x2==2:
        return -1
    elif x1==0 and x2==3:
        return 1
    elif x1==0 and x2==4:
        return -1
    elif x1==1 and x2==2:
        return 1
    elif x1==1 and x2==3:
        return -1
    elif x1==1 and x2==4:
        return 1
    elif x1==2 and x2==3:
        return 1
    elif x1==2 and x2==4:
        return -1
    elif x1==3 and x2==4:
        return 1
    elif x1==1 and x2==0:
        return -1
    elif x1==2 and x2==0:
        return 1
    elif x1==2 and x2==1:
        return -1
    elif x1==3 and x2==0:
        return -1
    elif x1==3 and x2==1:
        return 1
    elif x1==3 and x2==2:
        return -1
    elif x1==4 and x2==0:
        return 1
    elif x1==4 and x2==1:
        return -1
    elif x1==4 and x2==2:
        return 1
    elif x1==4 and x2==3:
        return -1

points_1=0
points_2=0


def winner(x,y):
    return "winner"

#########################################################-----ML
import pandas
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

featureX = pandas.read_csv("featureX.csv")
y = pandas.read_csv("featureY.csv")

predictors=[]
for i in range(1,2501):
    predictors.append(str(i))

#alg = RandomForestClassifier(random_state=1, n_estimators=25, min_samples_split=8, min_samples_leaf=4)
alg = LogisticRegression(random_state=1)
predictions=[]
kf = KFold(featureX.shape[0], n_folds=30, random_state=1)

for train, test in kf:
    #print "gain"
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (featureX[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = y["result"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(featureX[predictors].iloc[test,:])
    predictions.append(test_predictions)

scores = cross_validation.cross_val_score(alg, featureX[predictors], y["result"], cv=kf)
print(scores.mean())

#####################################################-----ML
lower_skin=np.array([50,100,90])
upper_skin=np.array([230,160,150])





vid1 = cv2.VideoCapture(0)
vid1.set(3,1280)
vid1.set(4,480)
ret, background=vid1.read()
#print background.shape
cv2.imshow('background',background)
vid1.release()

cap =  cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,480)
while(1):

    ret,frame = cap.read()
    frgd=frame-background
    cv2.imshow('frgd',frgd)

    thresh=cv2.cvtColor(frgd,cv2.COLOR_BGR2YCR_CB)
    #thresh=cv2.medianBlur(fgray,5)
    cv2.imshow('thresh',thresh)
    #plt.imshow(thresh,cmap='gray',interpolation='bicubic')
    #plt.show()
    kernel=np.array([5,5])
    thresh_c=cv2.inRange(thresh,lower_skin,upper_skin)
    thresh_c=cv2.erode(thresh_c,kernel,iterations=2)
    #ret,thresh=cv2.threshold(fgray,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('thresh_c',thresh_c)

    kernel=np.array([5,5])
    #thresh_c=cv2.erode(thresh,kernel,iterations=3)
    #thresh_c=cv2.medianBlur(thresh_c,5)
    #cv2.imshow('thresh_c',thresh_c)

    #frame_c=cv2.bitwise_and(frame,frame,mask=thresh_c)
    #frame_c=cv2.cvtColor(frame_c,cv2.COLOR_BGR2HSV)
    #plt.imshow(frame_c,cmap='gray',interpolation='bicubic')
    #plt.show()
    #
    # frme_c=cv2.medianBlur(frame_c,5)
    # plt.imshow(frame_c,cmap='gray',interpolation='bicubic')
    # plt.show()
    # frame_c=cv2.inRange(frame_c,lower_skin,upper_skin)
    # frame_c=cv2.dilate(frame_c,kernel,iterations=7)
    # frame_c=cv2.medianBlur(frame_c,15)
    # cv2.imshow('frame_c',frame_c)




    blank_mask=np.zeros(thresh_c.shape,np.uint8)
    trash = thresh_c.copy()
    contours, hierarchy = cv2.findContours(trash,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        #for cnt in contours:
            #print cv2.contourArea(cnt)
            largest_area=sorted(contours,key=cv2.contourArea)
        #cnt=largest_area[-1]
        #print cv2.contourArea(cnt)
    #emp=[]
            #if cv2.contourArea(cnt)>2000 and cv2.contourArea(cnt)<10000:
                #print "hi"
            if cv2.contourArea(largest_area[-1])>2000:
                cv2.drawContours(blank_mask,[largest_area[-1]],0,255,-1)
            if len(largest_area)>1:
                if cv2.contourArea(largest_area[-2])>2000:
                    cv2.drawContours(blank_mask,[largest_area[-2]],0,255,-1)
                #cnt=largest_area[-1]
    #     #if len(contours)>0:
    #         #cnt=max(contours,key=cv2.contourArea)
    #         #print cv2.contourArea(cnt)
    cv2.imshow('new_mask',blank_mask)


    # contours, hierarchy = cv2.findContours(blank_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     for cnt in contours:
    #         hull = cv2.convexHull(cnt,returnPoints=False)
    #         #if not hull is None:
    # # #
    #      #cv2.drawContours(frame,hull,0,(0,255,0),3)
    #         if not hull is None:
    #             defects = cv2.convexityDefects(cnt,hull)
    # # #     #emp=[]
    #             if not defects is None:
    #                 for i in range(defects.shape[0]):
    #                     s,e,f,d = defects[i,0]
    #                     start = tuple(cnt[s][0])
    #                     end = tuple(cnt[e][0])
    #                     far = tuple(cnt[f][0])
    #              #emp.append(far)
    #
    #                     cv2.line(frame,start,end,[0,255,0],2)
    #              #print far
    #
    #              #emp.append(far)
    #                     cv2.circle(frame,far,5,[0,0,255],-1)

    cv2.imshow('new_mask',blank_mask)
    cv2.imshow('thresh__c',frame)
    blanker=cv2.erode(blank_mask,kernel,iterations=3)
    blanker=cv2.medianBlur(blanker,5)
    cv2.imshow('blanker',blanker)

    #x,y=blanker.shape[:2]
    #print blanker.shape[1]
    half_1=blanker[0:blanker.shape[0],0:blanker.shape[1]/2]
    half_2=blanker[0:blanker.shape[0],blanker.shape[1]/2:blanker.shape[1]]
    cv2.imshow('aa',half_1)
    cv2.imshow('bb',half_2)


    ans1=half_1
    ans2=half_2
    contours1,_=cv2.findContours(half_1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours1)>0:
        cnt1=max(contours1,key=cv2.contourArea)
        x1,y1,w1,h1=cv2.boundingRect(cnt1)
        #print "11",x1,y1,w1,h1
        ans1=half_1[y1:y1+h1,x1:x1+w1]
        #ans1=cv2.medianBlur(ans1,5)
        cv2.imshow('ans1',ans1)
    contours2,_=cv2.findContours(half_2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours2)>0:
        cnt2=max(contours2,key=cv2.contourArea)
        x2,y2,w2,h2=cv2.boundingRect(cnt2)
        #print "22",x2,y2,w2,h2
        ans2=half_2[y2:y2+h2,x2:x2+w2]
        ans2=cv2.dilate(ans2,kernel,iterations=2)
        cv2.imshow('ans2',ans2)

    # if len(contours)>0 and len(contours)<2:
    #     cnt1=contours[0]
    #     x1,y1,w1,h1=cv2.boundingRect(cnt1)
    #     ans1=blanker[y1:y1+h1,x1:x1+w1]
    #     cv2.imshow('ans1',ans1)
    # if len(contours)>1:
    #     cnt1=contours[0]
    #     M1=cv2.moments(cnt1)
    #     cX1=1
    #     cX2=1
    #     if M1["m00"] != 0:
    #         cX1=int(M1["m10"] / M1["m00"])
    #     cnt2=contours[1]
    #     M2=cv2.moments(cnt2)
    #     if M2["m00"] != 0:
    #         cX2=int(M2["m10"] / M2["m00"])
    #     print "center1: " + str(cX1)
    #     print "center2: " + str(cX2)
    #     if(cX1>=cX2):
    #         cnt1=contours[0]
    #         cnt2=contours[1]
    #
    #     x1,y1,w1,h1=cv2.boundingRect(cnt1)
    #     ans1=blanker[y1:y1+h1,x1:x1+w1]
    #     cv2.imshow('ans1',ans1)
    #     cnt2=contours[1]
    #     x2,y2,w2,h2=cv2.boundingRect(cnt2)
    #     ans2=blanker[y2:y2+h2,x2:x2+w2]
    #     cv2.imshow('ans2',ans2)

    ans1=cv2.flip(ans1,1)
    cv2.imshow('ans1_f',ans1)
    mask_1 = cv2.resize(ans1, (50, 50))
    #mask_out=cv2.inRange(mask_out,lower,upper)
    mask_1=np.array(mask_1)
    mask_1=mask_1.ravel()
    mask_1=mask_1/255



    mask_2 = cv2.resize(ans2, (50, 50))
    #mask_out=cv2.inRange(mask_out,lower,upper)
    mask_2=np.array(mask_2)
    mask_2=mask_2.ravel()
    mask_2=mask_2/255



    final1=alg.predict(mask_1)
    final2=alg.predict(mask_2)



    print determiner(final1[0]),determiner(final2[0])

    k=cv2.waitKey(1)

    if k==ord('a'):
        winner=win_or_lose(final1[0],final2[0])
        if winner==1:
            points_2=points_2+1
        if winner==-1:
            points_1=points_1+1

    cv2.putText(frame,"Points:"+str(points_1),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,255)
    cv2.putText(frame,determiner(final1[0]),(100,620),cv2.FONT_HERSHEY_SIMPLEX,2,255)
    cv2.putText(frame,"Points:"+str(points_1),(900,100),cv2.FONT_HERSHEY_SIMPLEX,2,255)
    cv2.putText(frame,determiner(final2[0]),(900,620),cv2.FONT_HERSHEY_SIMPLEX,2,255)
    cv2.imshow('frm',frame)





    if k==ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()