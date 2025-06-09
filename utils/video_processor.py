import cv2
import tempfile # For creating temporary files
import os       # For path operations and deleting temp files

def get_video_source(source_type, uploaded_file=None):
    """
    Initializes a cv2.VideoCapture object based on source type.
    Args:
        source_type (str): "อัปโหลดไฟล์วิดีโอ" or "ใช้กล้องเว็บแคม".
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile, optional):
                      The uploaded file object from st.file_uploader.
    Returns:
        cv2.VideoCapture: The video capture object, or None if an error occurs.
    """
    cap = None
    temp_file_path = None

    if source_type == "ใช้กล้องเว็บแคม":
        try:
            cap = cv2.VideoCapture(0) # 0 for default webcam
            if not cap.isOpened():
                raise IOError("Cannot open webcam.")
            return cap, None # Return cap and no temp file path
        except Exception as e:
            print(f"Error opening webcam: {e}")
            return None, None

    elif source_type == "อัปโหลดไฟล์วิดีโอ":
        if uploaded_file is not None:
            try:
                # Create a temporary file to save the uploaded video
                # tempfile.NamedTemporaryFile creates a file that is automatically deleted when closed
                # or when the program exits.
                # 'delete=False' so we can close and then reopen with cv2.VideoCapture
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                print(f"Temporary video file saved at: {temp_file_path}")
                cap = cv2.VideoCapture(temp_file_path)
                
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file from temporary path: {temp_file_path}")
                return cap, temp_file_path # Return cap and the path to the temp file
            except Exception as e:
                print(f"Error processing uploaded video: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path) # Clean up temp file if error occurs
                return None, None
        else:
            print("No uploaded file provided.")
            return None, None
    
    return None, None # Should not reach here

def release_video_source(cap, temp_file_path=None):
    """
    Releases the cv2.VideoCapture object and cleans up temporary files.
    Args:
        cap (cv2.VideoCapture): The video capture object.
        temp_file_path (str, optional): Path to a temporary file to delete.
    """
    if cap:
        cap.release()
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
            print(f"Temporary file '{temp_file_path}' deleted.")
        except OSError as e:
            print(f"Error deleting temporary file {temp_file_path}: {e}")

# If you have other utility functions in this file, keep them here.
# For example, mirror_frame function:
def mirror_frame(frame):
    """Mirrors the frame horizontally."""
    return cv2.flip(frame, 1)