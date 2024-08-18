from fer import FER
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib
import time

# Set the backend for matplotlib to 'TkAgg' for compatibility with different environments
matplotlib.use('TkAgg')

# Initialize the FER (Face Emotion Recognition) detector using MTCNN
detector = FER(mtcnn=True)

# Start capturing video from the webcam (device 0)
cap = cv2.VideoCapture(0)

# Set a frame rate for recording the video (adjust based on your webcam's capabilities)
frame_rate = 4.3

# Initialize OpenCV's VideoWriter to save the video with the specified frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_video.avi', fourcc, frame_rate, (640, 480))

# Set up a matplotlib figure for displaying live emotion detection results
plt.ion()  # Turn on interactive mode for live updates
fig, ax = plt.subplots()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
bars = ax.bar(emotion_labels, [0]*7, color='lightblue') # Initialize bars for each emotion
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Real-time Emotion Detection')
ax.set_xticklabels(emotion_labels, rotation=45)

# Initialize imageio writer to save live chart updates as a GIF
gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)

# List to store cumulative emotion statistics for each frame
emotion_statistics = []

# Function to update the live chart
def update_chart(detected_emotions, bars, ax, fig):
    ax.clear()
    ax.bar(emotion_labels, [detected_emotions.get(emotion, 0) for emotion in emotion_labels], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Real-time Emotion Detection')
    ax.set_xticklabels(emotion_labels, rotation=45)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Start the timer to measure the active time of the webcam
webcam_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break  # Break the loop if no frame is captured

        # Get the dimensions of the frame
        height, width, channels = frame.shape

        # Detect emotions on the frame
        result = detector.detect_emotions(frame)
        largest_face = None
        max_area = 0

        # Find the largest face in the frame for primary emotion analysis
        for face in result:
            box = face["box"]
            x, y, w, h = box
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = face

        # If a face is detected, display the emotion and update the chart
        if largest_face:
            box = largest_face["box"]
            current_emotions = largest_face["emotions"]

            # Store the emotion data
            emotion_statistics.append(current_emotions)

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            emotion_type = max(current_emotions, key=current_emotions.get)
            emotion_score = current_emotions[emotion_type]

            emotion_text = f"{emotion_type}: {emotion_score:.2f}"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            update_chart(current_emotions, bars, ax, fig)

            out.write(frame)  # Write the frame to the video file

            # Save the current state of the bar chart as a frame in the GIF
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif_writer.append_data(image)

        cv2.imshow('Emotion Detection', frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    webcam_end_time = time.time()  # End timer when webcam window closes
    print(f"Webcam active time: {webcam_end_time - webcam_start_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)

    out.release()
    gif_writer.close()

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
