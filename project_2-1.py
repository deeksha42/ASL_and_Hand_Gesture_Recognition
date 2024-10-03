import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


word=set()

cap = cv2.VideoCapture(0)
dectector = HandDetector(maxHands=1)
classifier=Classifier("D:\Programs\Sacred_Eye\project\Model\keras_model.h5","D:\Programs\Sacred_Eye\project\Model\labels.txt")

offset = 20
imgsize = 300

folder = "Data/A" 
counter=0 
labels =["A","B","C","D","E","H","M","O","Okay","Peace","R","Smile","Stop","Thumbs Up","Thumbs Down","W","Y","I"]

while True:
    sucess , img= cap.read()
    imgOutput = img.copy()
    hands , img=dectector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3), np.uint8)

        imgcrop = img[y-offset:y+h+offset ,x-offset:x+w+offset]

        imgCropShape = imgcrop.shape

        aspectRatio = h/w

        if aspectRatio >1:
            k=imgsize/h
            wCal = math.ceil(k*w)
            imgResize= cv2.resize(imgcrop,(wCal,imgsize))
            imgResizeShpe = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
            prediction,index=classifier.getPrediction(imgWhite)
            #print(prediction,index)
            if index==12:
                print(str(word))
                break
            else:
                word.add(labels [index])


        
        else:
            k=imgsize/w
            hCal = math.ceil(k*h)
            imgResize= cv2.resize(imgcrop,(imgsize,hCal))
            imgResizeShpe = imgResize.shape
            hGap = math.ceil((imgsize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:] = imgResize
            prediction,index=classifier.getPrediction(imgWhite)
            #print(prediction,index)
            if index==12:
                print(str(word))
                break
                
            else:
                word.add(labels[index])


        

        cv2.putText(imgOutput,labels[index],(x,y-20), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        cv2.imshow("imageCrop",imgcrop)
        cv2.imshow("Imagewhite",imgWhite)


    cv2.imshow("Image",imgOutput)
    if cv2.waitKey(1)==27:
        break
print(str(word))
cap.release()
cv2.destroyAllWindows()
    