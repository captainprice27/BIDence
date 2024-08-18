import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from fer import FER
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import time

# Paths
path = "images_attendance"  # Folder containing images of students
attendance_file = "attendance.csv"  # CSV file to record attendance

# Initialization
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Encoding Faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Mark Attendance
def markAttendance(name, period):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{period},{dtString}")

# Initialize FER Detector
detector = FER(mtcnn=True)

# Initialize matplotlib for emotion detection chart
plt.ion()
fig, ax = plt.subplots()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
bars = ax.bar(emotion_labels, [0]*7, color='lightblue')
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Real-time Emotion Detection')
ax.set_xticklabels(emotion_labels, rotation=45)
gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)
emotion_statistics = []

# Start video capture
cap = cv2.VideoCapture(0)
encodeListKnown = findEncodings(images)
print("Encoding complete")

# Update live emotion chart
def update_chart(detected_emotions, bars, ax, fig):
    ax.clear()
    ax.bar(emotion_labels, [detected_emotions.get(emotion, 0) for emotion in emotion_labels], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Real-time Emotion Detection')
    ax.set_xticklabels(emotion_labels, rotation45)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Video capture and processing loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurrFrame = face_recognition.face_locations(imgS)
        encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

        # Facial Recognition and Attendance
        for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name, "Period 1")  # Customize "Period 1" as needed

        # Emotion Detection
        result = detector.detect_emotions(frame)
        largest_face = None
        max_area = 0

        for face in result:
            box = face["box"]
            x, y, w, h = box
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = face

        if largest_face:
            box = largest_face["box"]
            current_emotions = largest_face["emotions"]
            emotion_statistics.append(current_emotions)

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            emotion_type = max(current_emotions, key=current_emotions.get)
            emotion_score = current_emotions[emotion_type]

            emotion_text = f"{emotion_type}: {emotion_score:.2f}"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            update_chart(current_emotions, bars, ax, fig)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif_writer.append_data(image)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)
    gif_writer.close()

    # Save cumulative emotion data
    emotion_df = pd.DataFrame(emotion_statistics)
    plt.figure(figsize=(10, 10))
    for emotion in emotion_labels:
        plt.plot(emotion_df[emotion].cumsum(), label=emotion)
    plt.title('Cumulative Emotion Statistics Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Cumulative Confidence')
    plt.legend()
    plt.savefig('cumulative_emotions.jpg')
    plt.close()

    # Save attendance CSV
    print(f"Attendance recorded in {attendance_file}")
