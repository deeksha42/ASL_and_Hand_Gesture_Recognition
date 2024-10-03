import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# word tracking
word = []

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("D:/Programs/Sacred_Eye/project/Model/keras_model.h5", "D:/Programs/Sacred_Eye/project/Model/labels.txt")

offset = 20
imgsize = 300

labels = ["A", "B", "C", "D", "E", "H", "M", "O", "Okay", "Peace", "R", "Smile", "Stop", "Thumbs Up", "Thumbs Down", "W", "Y", "I"]

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgHeight, imgWidth, _ = img.shape
        if y - offset < 0 or y + h + offset > imgHeight or x - offset < 0 or x + w + offset > imgWidth:
            print("Hand out of frame bounds")
            continue

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        if imgcrop.size == 0:
            print("Invalid crop size")
            continue

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgsize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgcrop, (wCal, imgsize))
                wGap = math.ceil((imgsize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgsize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgcrop, (imgsize, hCal))
                hGap = math.ceil((imgsize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite,verbose=0)
        except Exception as e:
            print(f"Classifier error: {e}")
            continue

        if index == 12:  # Exit condition for 'Stop'
            print(str(word))
            break
        else:
            if labels[index] not in word:
                word.append(labels[index])

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.imshow("imageCrop", imgcrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) == 27:  # Esc to exit
        break

print(str(word))
cap.release()
cv2.destroyAllWindows()
