from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import base64
import io
from PIL import Image
import time

app = Flask(__name__)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Thresholds
thresh_ear = 0.25
thresh_mar = 1.5
frame_check = 20
yawn_limit = 2
drowsiness_limit = 5

flag = 0
yawn_count = 0
start_time = None
drowsiness_time = 0

# EAR / MAR functions same as before...
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    D = distance.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (2.0 * D)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global flag, yawn_count, start_time, drowsiness_time

    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = img.shape
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [61, 291, 78, 308, 13, 14, 17, 0]

            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
            mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            if ear < thresh_ear:
                flag += 1
                if flag >= frame_check:
                    if start_time is None:
                        start_time = time.time()
                    else:
                        drowsiness_time = time.time() - start_time
                    if drowsiness_time >= drowsiness_limit:
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                flag = 0
                start_time = None
                drowsiness_time = 0

            if mar >= thresh_mar:
                yawn_count += 1
                if yawn_count >= yawn_limit:
                    cv2.putText(img, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                yawn_count = 0

            cv2.putText(img, f"EAR: {ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5055)
