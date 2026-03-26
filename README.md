This project is a real‑time facial emotion scanner written in Python. It uses OpenCV to capture video from the webcam and the FER (Facial Emotion Recognition) library with the MTCNN model to detect faces and recognize seven basic emotions (angry, disgust, fear, happy, sad, surprise, neutral).

The program processes each frame in a background thread to keep the UI responsive (≈30 FPS). For every detected face it draws a colored rectangle based on the dominant emotion, displays the emotion name (translated to Spanish) with confidence percentage, and shows a small bar chart of confidence levels for all emotions.

Run python faceCAM.py to start the camera; press ‘q’ or Esc to exit the application.
