# app.py
from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import time

app = Flask(__name__)

# Thresholds
thresh_ear = 0.25
thresh_mar = 1.5
frame_check = 20
yawn_limit = 2
drowsiness_limit = 5

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    D = distance.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (2.0 * D)

# State variables
flag = 0
yawn_count = 0
start_time = None
drowsiness_time = 0

cap = cv2.VideoCapture(0)

def gen_frames():
    global flag, yawn_count, start_time, drowsiness_time

    while True:
        success, img = cap.read()
        if not success:
            break

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                mouth_indices = [61, 291, 78, 308, 13, 14, 17, 0]

                h, w, _ = img.shape
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
                mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth)

                cv2.polylines(img, [np.array(left_eye)], True, (0, 255, 0), 1)
                cv2.polylines(img, [np.array(right_eye)], True, (0, 255, 0), 1)
                cv2.polylines(img, [np.array(mouth)], True, (255, 0, 0), 1)

                if ear < thresh_ear:
                    flag += 1
                    cv2.putText(img, "CLOSED EYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if flag >= frame_check:
                        if start_time is None:
                            start_time = time.time()
                        else:
                            drowsiness_time = time.time() - start_time
                        if drowsiness_time >= drowsiness_limit:
                            cv2.putText(img, "DROWSINESS ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    flag = 0
                    start_time = None
                    drowsiness_time = 0
                    cv2.putText(img, "OPEN EYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if mar >= thresh_mar:
                    yawn_count += 1
                    cv2.putText(img, "YAWNING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if yawn_count >= yawn_limit:
                        cv2.putText(img, "YAWN ALERT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    yawn_count = 0

                cv2.putText(img, f"EAR: {ear:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"MAR: {mar:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5055)

