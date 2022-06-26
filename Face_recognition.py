import os
import cv2
import numpy as np


# The Knn
def distx(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(train,test,k=5):

    dist=[]
    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i, -1]
        d=distx(ix,test)
        dist.append([d,iy])
    
    dk = sorted(dist, key=lambda x:x[0])
    labels = np.array(dk[:k])[:,-1]

    #majority vote
    output =np.unique(labels,return_counts=True)
    indx = output[1].argmax()
    return output[0][indx]


skip = 0
dataPath = "data/"

faceData = []
labels = []

classId = 0
names = {}

# Data Preparation
for file in os.listdir(dataPath):
    if file.endswith(".npy"):
        names[classId] = file[:-4]
        data = np.load(dataPath+file)
        faceData.append(data)
        
        #print(data.shape)

        dataLabel = classId*np.ones(data.shape[0],)
        classId +=1
        labels.append(dataLabel)


FaceDataSet = np.concatenate(faceData, axis=0)
FaceLabels = np.concatenate(labels, axis=0)

# print(FaceDataSet.shape)
# print(FaceLabels.shape)

FaceLabels = FaceLabels.reshape(-1,1)
# print(FaceLabels.shape)

trainData = np.concatenate((FaceDataSet, FaceLabels), axis=1)

#print(trainData.shape)

cap=cv2.VideoCapture(0) # 0 = default webcam
faceCascade=cv2.CascadeClassifier("haarcascade_fontalface_alt.xml")

while True:

    capture,frame = cap.read()

    if not capture:
        continue

    # grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # faceCascade.detectMultiScale(grayFrame,scalingFactor,NoOfNeigbours)
    # scalingFactor shrink the image
    # 1.3 shrink 30% of every pass, similarly 1.05 shrink 5%
    faces = faceCascade.detectMultiScale(frame,1.3,5)


    for face in faces:
        (x,y,w,h) = face

        offset = 10
        faceSection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        faceSection = cv2.resize(faceSection,(150,150))


        pred = knn(trainData, faceSection.flatten())
        predName = names[pred]

        cv2.putText(frame,predName,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Faces",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

