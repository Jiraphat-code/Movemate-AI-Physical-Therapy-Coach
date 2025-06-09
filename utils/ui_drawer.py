import cv2
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose

def to_pixel_coords(point, width, height):
    """Converts normalized (x,y) to pixel coordinates."""
    return tuple(np.multiply(point, [width, height]).astype(int))

def draw_ui(image, counter, stage):
    """Draws REPS and STAGE UI on the image."""
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(stage), (70,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    return image

def draw_class_probabilities(image, class_names, body_language_prob, width, height):
    """Draws class probabilities on the image."""
    margin = 30
    text_height_spacing = 25 # Increased spacing for readability
    for idx, (cls, prob) in enumerate(zip(class_names, body_language_prob)):
        all_class_prob = f"{cls}: {prob:.2f}"
        # (text_width, _), _ = cv2.getTextSize(all_class_prob, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1) # Calculate text width dynamically

        # Fixed position for now, adjust if needed
        position = (image.shape[1] - 200, margin + idx * text_height_spacing) # Approx. 200 pixels from right edge
        cv2.putText(image, all_class_prob, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    return image

def draw_prediction_on_ear(image, results, body_language_class, body_language_prob):
    """Draws predicted class and probability near the ear."""
    if results.pose_landmarks:
        try:
            coords = to_pixel_coords(
                np.array((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y)),
                image.shape[1], image.shape[0] # Pass width and height
            )
            prob_value = round(body_language_prob[np.argmax(body_language_prob)], 2)
            display_class_prob = f"{body_language_class} ({prob_value})"

            (text_width, text_height_val), _ = cv2.getTextSize(display_class_prob, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Adjusted rect position for better visibility
            rect_start = (coords[0], coords[1] - text_height_val - 5)
            rect_end = (coords[0] + text_width + 10, coords[1] + 5)

            cv2.rectangle(image, rect_start, rect_end, (245, 117, 16), -1)
            cv2.putText(image, display_class_prob, (coords[0] + 5, coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            # Handle cases where ear landmark might not be detected
            # print(f"Could not draw prediction on ear: {e}")
            pass
    return image