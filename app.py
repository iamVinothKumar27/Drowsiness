import streamlit as st
import cv2
import numpy as np
import time
from pygame import mixer
import os

# Initialize pygame.mixer only if not running in Streamlit Cloud
try:
    mixer.init()
    AUDIO_ENABLED = True
except:
    AUDIO_ENABLED = False

# Streamlit UI
st.title("Real-time Drowsiness Detection")
st.write("Click the button below to start detection:")

# Load your model or cascade (you can replace this with your actual model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Dummy function to simulate alert
def play_alert():
    if AUDIO_ENABLED:
        mixer.music.load("alert.wav")  # Make sure this file is available
        mixer.music.play()
    else:
        st.error("⚠️ Drowsiness detected! (Audio alert disabled on Streamlit Cloud)")

# Detection logic
def detect_drowsiness():
    cap = cv2.VideoCapture(0)  # Use webcam
    score = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        eyes = eye_cascade.detectMultiScale(gray)

        # Drowsiness logic
        if len(eyes) == 0:
            score += 1
        else:
            score = max(score - 1, 0)

        # Show the frame (optional)
        cv2.imshow('Drowsiness Detection', frame)

        # Trigger alert
        if score > 10:
            play_alert()
            break

        if cv2.waitKey(1) == ord('q') or time.time() - start_time > 15:  # Timeout
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit interaction
if st.button("Start Detection"):
    st.write("Detection started. Please stay visible to the camera...")
    detect_drowsiness()
