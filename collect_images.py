import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
dectector = HandDetector(maxHands=1)

offset = 20
imgsize = 300

folder = r"D:\Programs\Sacred_Eye\project\Data\Smile"
counter=0 

while True:
    sucess, img= cap.read()
    hands, img = dectector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3), np.uint8)*255

        imgcrop = img[y-offset:y+h+offset ,x-offset:x+w+offset]

        imgCropShape = imgcrop.shape


        aspectRatio = h/w

        if aspectRatio >1:
            k=imgsize/h
            wCal = math.ceil(k*w)
            imgResize= cv2.resize(imgcrop, (wCal,imgsize))
            imgResizeShpe = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
        
        else:
            k=imgsize/w 
            w
            hCal = math.ceil(k*h)
            imgResize= cv2.resize(imgcrop,(imgsize,hCal))
            imgResizeShpe = imgResize.shape
            hGap = math.ceil((imgsize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:] = imgResize
        


        cv2.imshow("imageCrop",imgcrop)
        cv2.imshow("Imagewhite",imgWhite)


    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter+=1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",imgWhite)
        print(counter)
    
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
