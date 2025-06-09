import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose

# Helper functions (เหมือนเดิม)
def euclidean_distance(x1, y1, x2, y2):
    """Calculates Euclidean distance between two 2D points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three 2D points (b is the vertex)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_landmark_xy(landmarks_list, landmark_enum, min_visibility=0.5):
    """Helper to get x,y coordinates from a list of MediaPipe landmarks by enum,
       returning None if landmark is not sufficiently visible."""
    landmark = landmarks_list[landmark_enum.value]
    if landmark.visibility > min_visibility:
        return [landmark.x, landmark.y]
    return None # Return None if not visible enough

def extract_features(landmarks, model_type="33_points"):
    """
    Extracts features based on the selected model type.
    landmarks: A list of MediaPipe pose landmarks (results.pose_landmarks.landmark).
    model_type: "33_points", "17_points", or "17_points_plus_angles".
    """
    if not landmarks:
        return None # Return None if no landmarks are provided

    # Always start with raw flatten (for 33 points)
    all_landmarks_flat = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()

    if model_type == "33_points":
        return pd.DataFrame([all_landmarks_flat])

    # Define the 17 keypoints (adjust based on your model's training)
    selected_17_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    
    # Extract only the selected 17 landmarks
    selected_landmarks_flat = np.array([
        [landmarks[idx.value].x, landmarks[idx.value].y, landmarks[idx.value].z, landmarks[idx.value].visibility]
        for idx in selected_17_indices
    ]).flatten()
    
    if model_type == "17_points":
        return pd.DataFrame([selected_landmarks_flat])

    elif model_type == "17_points_plus_angles":
        features_with_angles = selected_landmarks_flat.tolist() # Convert to list to append new features

        # --- Extract XY Coordinates for Calculation with visibility check ---
        ls = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        le = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        lw = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        lh = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        ear_l = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_EAR) # Now it returns None if not visible

        # --- Calculate 6 Additional Features, handle missing landmarks ---
        
        # Initialize angles/distances to a default (e.g., 0.0 or np.nan) if landmarks are missing
        dist_shoulder_elbow = 0.0
        dist_elbow_wrist = 0.0
        dist_shoulder_hip = 0.0
        angle_elbow = 0.0
        angle_torso = 0.0
        angle_shoulder = 0.0

        # Distances
        if ls and le:
            dist_shoulder_elbow = euclidean_distance(*ls, *le)
        if le and lw:
            dist_elbow_wrist = euclidean_distance(*le, *lw)
        if ls and lh:
            dist_shoulder_hip = euclidean_distance(*ls, *lh)

        # Angles
        if ls and le and lw:
            angle_elbow = calculate_angle(ls, le, lw)
        if lh and ls and ear_l: # Ensure all 3 points are visible
            angle_torso = calculate_angle(lh, ls, ear_l)
        if lh and ls and le:
            angle_shoulder = calculate_angle(lh, ls, le)
        
        features_with_angles.extend([
            dist_shoulder_elbow,
            dist_elbow_wrist,
            dist_shoulder_hip,
            angle_elbow,
            angle_torso,
            angle_shoulder
        ])
        
        return pd.DataFrame([features_with_angles])

    return None # If model_type is invalid