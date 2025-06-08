import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import pickle
import time
from rep_classifier import classify_reps

# Load model
with open('models\movemate_no_encoding_rf.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('AI PT Coach - Video Analysis')

uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'avi'])

# --- Functions from main.py for visualization ---
def to_pixel_coords(point, width=640, height=480):
    return tuple(np.multiply(point, [width, height]).astype(int))

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

def draw_ui(image, counter, stage):
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(stage), (70,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

def draw_class_probabilities(image, class_names, body_language_prob):
    margin = 30
    text_height = 20
    for idx, (cls, prob) in enumerate(zip(class_names, body_language_prob)):
        all_class_prob = f"{cls}: {prob:.2f}"
        (text_width, _), _ = cv2.getTextSize(all_class_prob, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        position = (image.shape[1] - text_width - margin, margin + idx * text_height)
        cv2.putText(image, all_class_prob, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

def process_frame(frame, pose, model, stage, current_rep_start, counter, frame_idx, rep_segments):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    body_language_class = None
    body_language_prob = None
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
            # Draw vectors
            cv2.line(image, to_pixel_coords(l_shoulder), to_pixel_coords(l_hip), (0, 255, 0), 10)
            cv2.line(image, to_pixel_coords(l_shoulder), to_pixel_coords(l_elbow), (0, 0, 255), 10)
            cv2.putText(image, str(l_angle), tuple(np.multiply(l_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
            # Rep logic
            if stage is None and l_angle < 30:
                stage = "Start!"
                current_rep_start = frame_idx
            if l_angle > 90:
                stage = "up"
            if l_angle < 30 and stage == 'up':
                stage = "down"
                counter += 1
                if current_rep_start is not None:
                    rep_segments.append((current_rep_start, frame_idx))
                current_rep_start = frame_idx + 1
            # Render pose
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            # Predict class
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
            X = pd.DataFrame([pose_row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            # Draw class+prob near ear
            coords = tuple(np.multiply(
                np.array((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y)), [640,480]).astype(int))
            prob_value = round(body_language_prob[np.argmax(body_language_prob)], 2)
            display_class_prob = f"{body_language_class} ({prob_value})"
            (text_width, text_height), _ = cv2.getTextSize(display_class_prob, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            rect_start = (coords[0], coords[1] + 5)
            rect_end = (coords[0] + text_width + 10, coords[1] - 30)
            cv2.rectangle(image, rect_start, rect_end, (245, 117, 16), -1)
            cv2.putText(image, display_class_prob, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Draw all class probabilities
            try:
                class_names = model.classes_
            except AttributeError:
                class_names = [f"Class{i+1}" for i in range(len(body_language_prob))]
            draw_class_probabilities(image, class_names, body_language_prob)
        except Exception as e:
            print("Error in prediction:", e)
    draw_ui(image, counter, stage)
    return image, stage, current_rep_start, counter, body_language_class

# --- End functions from main.py ---

if uploaded_file is not None:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Save uploaded file to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # OpenCV video capture
    cap = cv2.VideoCapture(video_path)
    frame_classifications = []
    rep_segments = []
    current_rep_start = None
    stage = None
    counter = 0
    frame_idx = 0

    # Create a placeholder for video frames
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            image, stage, current_rep_start, counter, body_language_class = process_frame(
                frame, pose, model, stage, current_rep_start, counter, frame_idx, rep_segments
            )
            frame_classifications.append(body_language_class)
            frame_idx += 1
            # Show the processed frame in Streamlit
            frame_placeholder.image(image, channels="RGB")
            time.sleep(0.03)  # Add a small delay for smoother playback
        cap.release()

    if rep_segments:
        rep_results, summary = classify_reps(frame_classifications, rep_segments)
        st.write(f"Total reps: {len(rep_results)}")
        st.write("Summary:")
        for cls, cnt in summary.items():
            st.write(f"{cnt} reps classified as {cls}")
    else:
        st.write("No reps detected.")
