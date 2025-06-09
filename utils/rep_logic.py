import numpy as np
from collections import Counter
import mediapipe as mp # Needed for PoseLandmark if any rep logic uses specific points

# Define the available classes (should match your model's output classes)
REP_CLASSES = ['elbow_bent', 'mix_abduction', 'neck_tilt', 'normal', 'trunk_twist']

# This function might be needed if calculate_angle is not imported elsewhere
# Or ensure it's imported from feature_extractor.py
def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three 2D points (b is the vertex)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def update_rep_counter(l_angle, current_stage, current_counter, frame_idx, current_rep_start, rep_segments):
    """
    Updates the repetition counter and stage based on the elbow angle.
    This is a simple example for bicep curl. Adjust thresholds for your exercise.
    """
    stage = current_stage
    counter = current_counter

    # Your original rep logic from provided code
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

    return stage, counter, current_rep_start, rep_segments


def classify_reps(frame_classifications, rep_segments, version=None, class_labels=None):
    """
    Analyzes frame classifications within detected repetition segments to classify each rep.
    Args:
        frame_classifications: List of class labels for each frame (e.g., ["normal", "elbow_bent", ...])
        rep_segments: List of (start_frame, end_frame) tuples for each rep (e.g., [(0, 9), (10, 19), ...])
        version: Optional string to specify versioning logic (future use)
        class_labels: Optional list of class labels to use (default: REP_CLASSES)
    Returns:
        rep_results: List of predicted class for each rep
        summary: Counter of class counts across all reps
    """
    if class_labels is None:
        class_labels = REP_CLASSES

    rep_results = []
    for start, end in rep_segments:
        # Ensure 'end' index does not go out of bounds for frame_classifications
        # And ensure proper slicing based on whether 'end' is inclusive or exclusive
        # The original code used end+1, implying end is inclusive.
        rep_classes = frame_classifications[start:min(end + 1, len(frame_classifications))]

        # Only consider classes in class_labels
        filtered_classes = [cls for cls in rep_classes if cls in class_labels]
        
        if filtered_classes:
            most_common_class, count = Counter(filtered_classes).most_common(1)[0]
        else:
            most_common_class = "unknown_class" # Handle cases where no recognized classes are found in a segment
        rep_results.append(most_common_class)

    summary = Counter(rep_results)
    return rep_results, summary