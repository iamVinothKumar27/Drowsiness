import cv2
import av
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import time

# EAR & MAR thresholds
thresh_ear = 0.25
thresh_mar = 1.5
frame_check = 20
yawn_limit = 2
drowsiness_limit = 5

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.flag = 0
        self.yawn_count = 0
        self.start_time = None
        self.drowsiness_time = 0
        self.show_drowsy_alert = False
        self.show_yawn_alert = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        self.show_drowsy_alert = False
        self.show_yawn_alert = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmarks for eyes and mouth
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                mouth_indices = [61, 291, 78, 308, 13, 14, 17, 0]

                h, w, _ = img.shape

                left_eye = [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
                right_eye = [(int(face_landmarks.landmark[i].x * w),
                              int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
                mouth = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth)

                cv2.polylines(img, [cv2.convexHull(np.array(left_eye))], True, (0, 255, 0), 1)
                cv2.polylines(img, [cv2.convexHull(np.array(right_eye))], True, (0, 255, 0), 1)
                cv2.polylines(img, [cv2.convexHull(np.array(mouth))], True, (255, 0, 0), 1)

                # Drowsiness detection
                if ear < thresh_ear:
                    self.flag += 1
                    if self.flag >= frame_check:
                        if self.start_time is None:
                            self.start_time = time.time()
                        else:
                            self.drowsiness_time = time.time() - self.start_time
                        if self.drowsiness_time >= drowsiness_limit:
                            self.show_drowsy_alert = True
                    cv2.putText(img, "CLOSED EYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.flag = 0
                    self.start_time = None
                    self.drowsiness_time = 0
                    cv2.putText(img, "OPEN EYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Yawning detection
                if mar >= thresh_mar:
                    self.yawn_count += 1
                    if self.yawn_count >= yawn_limit:
                        self.show_yawn_alert = True
                    cv2.putText(img, "YAWNING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    self.yawn_count = 0

                cv2.putText(img, f"EAR: {ear:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"MAR: {mar:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img

st.set_page_config(page_title="Drowsiness Detector")
st.title("🛡 Real-Time Drowsiness Detection")

ctx = webrtc_streamer(key="key", video_transformer_factory=VideoTransformer)

# Display alert messages in sidebar
if ctx.video_transformer:
    if ctx.video_transformer.show_drowsy_alert:
        st.sidebar.error("😴 Drowsiness Detected!")
    elif ctx.video_transformer.show_yawn_alert:
        st.sidebar.warning("😮 Yawning Detected!")
    else:
        st.sidebar.success("✅ Normal Status")
