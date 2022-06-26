from fileinput import filename
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture(0) # 0 = default webcam
faceCascade=cv2.CascadeClassifier("haarcascade_fontalface_alt.xml")

skip = 0
faceData = []

dataSetPath = 'data/'
fileName = input("Enter the name of Person : ")

while True:

    capture,frame = cap.read()

    if not capture:
        continue

    # grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # faceCascade.detectMultiScale(grayFrame,scalingFactor,NoOfNeigbours)
    # scalingFactor shrink the image
    # 1.3 shrink 30% of every pass, similarly 1.05 shrink 5%
    faces = faceCascade.detectMultiScale(frame,1.5,5)
    faces = sorted(faces, key=lambda x:x[2]*x[3],reverse=True)

    faceSection = 0
    for face in faces:
        (x,y,w,h) = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        offset = 10
        faceSection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        faceSection = cv2.resize(faceSection,(150,150))

        if skip%10 == 0:
            faceData.append(faceSection)
            print(len(faceData))
        skip+=1

    cv2.imshow("Video Frame",frame)
    cv2.imshow('Face Section',faceSection)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


faceData=np.asarray(faceData)
faceData = faceData.reshape((faceData.shape[0],-1))
print(faceData.shape)

np.save(dataSetPath+fileName+'.npy',faceData)
print("Data Successfully save at "+ dataSetPath+fileName+'.npy')

cap.release()
cv2.destroyAllWindows()

