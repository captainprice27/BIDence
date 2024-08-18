import cv2
#import face_recognition
import numpy as np
import os
from datetime import datetime

# 1.Extraction of images from file.

path = "images_attendance"
images = []  # This will store the images.
classNames = []  # This will store the names of the student.
myList = os.listdir(path)  # this will extract all the images from the path and will store the path to that image
print(myList)

# 2.Creation of list of students in class.

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# 3.Start with encoding process


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name}, {dtString}")

encodeListKnown = findEncodings(images)
print("Encoding complete")

# 4. Find matches between our encoding

cap = cv2.VideoCapture(0)

while True:  #To get each frame one by one
    success, img = cap.read()
    # Because we are doing this in real time we will want to reduce our time because this will help in speeding up
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # In the webcam we can get multiple faces to get the position of faces we will use
    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)




