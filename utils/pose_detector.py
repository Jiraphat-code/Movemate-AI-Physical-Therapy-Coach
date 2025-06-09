import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose_model = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image):
        """
        Performs pose detection on a single image frame.
        Returns the image with processing flags reset and the MediaPipe results object.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        """
        Draws pose landmarks and connections on the image.
        """
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        return image

    def extract_landmarks(self, results):
        """
        Extracts raw landmark coordinates from MediaPipe results.
        Returns a flattened list of [x, y, z, visibility] for each landmark, or None if no landmarks found.
        """
        if results.pose_landmarks:
            return np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in results.pose_landmarks.landmark]).flatten()
        return None
