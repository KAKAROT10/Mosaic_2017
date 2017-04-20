import cv2
import numpy as np

lower=np.array([10,10,10])
upper=np.array([255,255,255])

for i in range(1,31):

    img=cv2.imread('spock'+str(i)+'.png',1)
    img_f=cv2.inRange(img,lower,upper)
    cv2.imwrite('spock_f'+str(i)+'.png',img_f)

cv2.waitKey(0)
cv2.destroyAllWindows()