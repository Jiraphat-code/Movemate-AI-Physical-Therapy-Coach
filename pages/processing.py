import streamlit as st
import cv2
import time
import numpy as np
import mediapipe as mp # Import here as well for clarity within the page's scope
from collections import Counter # Need Counter for classify_reps summary
import os
import pandas as pd

# Import functions from our utils package
from utils.pose_detector import PoseDetector
from utils.feature_extractor import extract_features, calculate_angle as feature_calc_angle # Rename if there's a conflict
from utils.model_loader import load_model
from utils.ui_drawer import draw_ui, draw_class_probabilities, draw_prediction_on_ear, to_pixel_coords
from utils.video_processor import get_video_source, release_video_source, mirror_frame
from utils.rep_logic import update_rep_counter, classify_reps, REP_CLASSES # Import classify_reps and REP_CLASSES

mp_pose = mp.solutions.pose # Need to redefine for use with Landmark enums

def app():
    st.title("📊 ประมวลผล & รายงานผล")
    st.markdown("---")

    # Check if necessary data from Feature Selection is available
    if 'video_source' not in st.session_state or \
       'selected_arm' not in st.session_state or \
       'selected_model_type' not in st.session_state or \
       'model_path' not in st.session_state:
        st.warning("กรุณาเลือกการตั้งค่าในหน้า 'เลือกการประมวลผล' ก่อน.")
        if st.button("กลับไปหน้าเลือกการประมวลผล"):
            st.session_state.current_page_option = "Feature Selection" # Correct page name for option_menu
            st.rerun()
        return # Stop execution if settings are not found

    video_source = st.session_state.video_source
    video_name = st.session_state.uploaded_video_name
    selected_arm = st.session_state.selected_arm
    selected_model_type = st.session_state.selected_model_type
    model_file_path = st.session_state.model_path

    st.write(f"**แหล่งที่มา:** {video_source} {video_name}")
    st.write(f"**แขนที่เลือก:** {selected_arm}")
    st.write(f"**โมเดลที่ใช้:** {selected_model_type} {model_file_path}")

    # Load the selected model
    model = load_model(model_file_path)
    if model is None:
        st.error("ไม่สามารถโหลดโมเดลได้. โปรดตรวจสอบไฟล์โมเดลและลองอีกครั้ง.")
        if st.button("กลับไปหน้าเลือกการประมวลผล"):
            st.session_state.current_page_option = "Feature Selection" # Correct page name
            st.rerun()
        return
    
    # Initialize MediaPipe Pose detector
    pose_detector = PoseDetector()

    # Streamlit video placeholder
    st_frame = st.empty()

    # Control buttons
    col1, col2 = st.columns(2)
    start_button = col1.button("เริ่มประมวลผล")
    stop_button = col2.button("หยุดประมวลผล")

    # Initial states for rep counting and classification
    counter = 0
    stage = None
    current_rep_start = None
    rep_segments = []            # List to store (start_frame, end_frame) for each rep
    frame_classifications = []   # List to store class label for each frame
    frame_idx = 0

    # Variable to store temporary file path
    temp_video_path = None # Initialize outside the if block

    if start_button:
        st.info("กำลังเริ่มประมวลผล...")
        # Get the uploaded_video_file object from session state
        uploaded_video_file_obj = st.session_state.get('uploaded_video_file', None)
        
        # Call get_video_source, now it returns cap and temp_file_path
        cap, temp_video_path = get_video_source(video_source, uploaded_video_file_obj)

        if cap is None:
            st.error("ไม่สามารถเข้าถึงแหล่งวิดีโอได้. โปรดตรวจสอบกล้องหรือไฟล์ที่อัปโหลด.")
            # Ensure temp_video_path is cleaned up if it was created before error
            if temp_video_path and os.path.exists(temp_video_path):
                 os.remove(temp_video_path)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_source == "อัปโหลดไฟล์วิดีโอ" else 0
        progress_bar = st.progress(0) if total_frames > 0 else None

        processing_active = True
        while cap.isOpened() and processing_active:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Mirroring Logic ---
            if selected_arm == "ขวา (Right Arm)":
                frame = mirror_frame(frame)

            # Get frame dimensions for drawing
            frame_height, frame_width, _ = frame.shape

            # 1. Pose Detection
            image, results = pose_detector.detect(frame)
            
            # Default values if no pose is detected or error
            body_language_class = "no_pose" # Default class if no pose is detected
            body_language_prob = np.zeros(len(model.classes_)) if hasattr(model, 'classes_') else np.zeros(len(REP_CLASSES))

            if results.pose_landmarks:
                # กำหนดค่าเริ่มต้นให้ features เป็น None ก่อนที่จะพยายามดึงค่า
                features = None 
                
                try:
                    # แปลงชื่อโมเดลที่เลือกให้เป็นรูปแบบที่ extract_features เข้าใจ
                    feature_type_str = ""
                    if selected_model_type == "โมเดล 33 จุด (Raw MediaPipe Landmarks)":
                        feature_type_str = "33_points"
                    elif selected_model_type == "โมเดล 17 จุด (Selected MediaPipe Landmarks)":
                        feature_type_str = "17_points"
                    elif selected_model_type == "โมเดล 17 จุด + Features (ระยะห่าง/มุม)":
                        feature_type_str = "17_points_plus_angles"
                    else:
                        st.warning(f"Model type '{selected_model_type}' not recognized by feature extractor.")
                        body_language_class = "unrecognized_model_type"
                        # features ยังคงเป็น None
                        
                    # พยายามดึง features ออกมา ถ้า feature_type_str ถูกต้อง
                    if feature_type_str: # Only attempt extraction if feature_type_str is valid
                        features = extract_features(results.pose_landmarks.landmark, model_type=feature_type_str)
                    
                    if features is not None and not features.empty:
                        # Predict class
                        body_language_class = model.predict(features)[0]
                        body_language_prob = model.predict_proba(features)[0]

                        # --- NEW CODE FOR SHOULDER ANGLE DISPLAY ---
                        try:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Determine which arm to display based on selected_arm
                            if selected_arm == "ซ้าย (Left Arm)":
                                target_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                target_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                target_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            elif selected_arm == "ขวา (Right Arm)":
                                # If mirroring is on, the "RIGHT" landmarks visually appear on the left side of the frame.
                                # But we still reference them as RIGHT_... in MediaPipe.
                                target_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                target_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                                target_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            else: # Default to Left Arm for display if "Both Arms" or unspecified
                                target_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                target_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                target_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                            # Ensure all points are detected and visible before calculating/drawing
                            if all(p is not None for p in [target_hip, target_shoulder, target_elbow]):
                                # Calculate the angle at the shoulder (Hip-Shoulder-Elbow)
                                shoulder_angle = feature_calc_angle(target_hip, target_shoulder, target_elbow)

                                # Draw vectors (Hip-Shoulder, Shoulder-Elbow)
                                # Using to_pixel_coords for drawing
                                cv2.line(image, to_pixel_coords(target_shoulder, frame_width, frame_height), 
                                                to_pixel_coords(target_hip, frame_width, frame_height), (0, 255, 0), 10) # Green for Hip-Shoulder
                                cv2.line(image, to_pixel_coords(target_shoulder, frame_width, frame_height), 
                                                to_pixel_coords(target_elbow, frame_width, frame_height), (0, 0, 255), 10) # Blue for Shoulder-Elbow
                                
                                # Display the shoulder angle near the shoulder
                                angle_display_coords = to_pixel_coords(target_shoulder, frame_width, frame_height)
                                cv2.putText(image, f"Sh. Angle: {int(shoulder_angle)}", 
                                            (angle_display_coords[0] + 20, angle_display_coords[1] - 20), # Offset text slightly
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # White text
                            else:
                                st.write("Warning: Some shoulder/hip/elbow landmarks are not visible for angle display.")
                        except Exception as angle_display_e:
                            # Catch specific errors if landmarks are not found for display
                            st.warning(f"Error displaying shoulder angle or vectors: {angle_display_e}")

                        # --- Repetition Counting Logic ---
                        angle_to_track = None
                        try:
                            # ... (ส่วนการคำนวณ angle_to_track เหมือนเดิม) ...
                            if selected_arm == "ซ้าย (Left Arm)":
                                l_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                l_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                                l_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                angle_to_track = feature_calc_angle(l_hip, l_shoulder, l_elbow)
                            elif selected_arm == "ขวา (Right Arm)":
                                r_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                                r_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                                r_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                angle_to_track = feature_calc_angle(r_hip, r_shoulder, r_elbow)
                            
                            if angle_to_track is not None:
                                stage, counter, current_rep_start, rep_segments = update_rep_counter(
                                    angle_to_track, stage, counter, frame_idx, current_rep_start, rep_segments
                                )
                        except Exception as angle_e:
                            st.warning(f"ไม่สามารถคำนวณมุมสำหรับการนับรอบได้: {angle_e}. อาจเป็นเพราะ Landmark ขาดหายไป.")
                            pass
                        
                    else: # ถ้า features เป็น None หรือ Empty DataFrame หลังจากการดึง
                        st.warning("ไม่สามารถดึงคุณลักษณะ (Features) จาก Landmark ได้ หรือ features ว่างเปล่า.")
                        body_language_class = "no_features" 
                        # body_language_prob ยังคงเป็นค่าเริ่มต้น (zeros)
                except Exception as e:
                    st.warning(f"เกิดข้อผิดพลาดในการประมวลผล AI: {e}")
                    body_language_class = "error_processing" 
            else: # ถ้า results.pose_landmarks ไม่มีข้อมูลเลย
                body_language_class = "no_pose" 
                # body_language_prob ยังคงเป็นค่าเริ่มต้น (zeros)

            frame_classifications.append(body_language_class)

            # 3. Drawing UI
            image = pose_detector.draw_landmarks(image, results)
            image = draw_ui(image, counter, stage)
            if results.pose_landmarks: # Only draw prediction if landmarks are present
                image = draw_prediction_on_ear(image, results, body_language_class, body_language_prob)
                try:
                    class_names_for_display = model.classes_ if hasattr(model, 'classes_') else REP_CLASSES # Use REP_CLASSES as fallback
                    image = draw_class_probabilities(image, class_names_for_display, body_language_prob, frame_width, frame_height)
                except Exception as e:
                    # print(f"Could not draw class probabilities: {e}")
                    pass 

            # Display the processed frame
            st_frame.image(image, channels="BGR", use_container_width=True)

            # Update progress bar
            if progress_bar:
                current_progress = int((frame_idx / total_frames) * 100)
                progress_bar.progress(current_progress)

            frame_idx += 1
            time.sleep(0.01) # Small delay to prevent overwhelming CPU/GPU

            # Check for stop button press inside the loop
            if stop_button: # This checks the button state on each rerun
                processing_active = False
                break # Exit the loop if stop button is pressed

        release_video_source(cap)
        st.success("การประมวลผลเสร็จสิ้น!")

        # --- ส่วนแสดงข้อมูลคนไข้ (ใหม่) ---
        st.subheader("📋 ข้อมูลคนไข้")
        patient_info = st.session_state.get('patient_info', {}) # ดึงข้อมูลคนไข้
        
        if patient_info:
            # ใช้ st.columns เพื่อจัดวางข้อมูลเป็น 2 คอลัมน์
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(f"**รหัสคนไข้:** `{patient_info.get('รหัสคนไข้', 'N/A')}`")
                st.markdown(f"**ชื่อ-นามสกุล:** `{patient_info.get('ชื่อ-นามสกุล', 'N/A')}`")
                st.markdown(f"**อายุ:** `{patient_info.get('อายุ', 'N/A')}` ปี")
            with col_p2:
                st.markdown(f"**เพศ:** `{patient_info.get('เพศ', 'N/A')}`")
                st.markdown(f"**เบอร์โทรศัพท์:** `{patient_info.get('เบอร์โทรศัพท์', 'N/A')}`")
            
            st.markdown(f"**การวินิจฉัย/อาการ:** `{patient_info.get('การวินิจฉัย/อาการ', 'N/A')}`")
        else:
            st.info("ยังไม่มีข้อมูลคนไข้ กรุณาลงทะเบียนในหน้า 'Register'.")
        st.markdown("---")
        
         # --- ส่วนแสดงผลลัพธ์โมเดล (ปรับปรุงใหม่) ---
        st.subheader("📊 สรุปผลการประเมินท่าทาง")
        
        if rep_segments:
            rep_results, summary = classify_reps(frame_classifications, rep_segments, class_labels=model.classes_ if hasattr(model, 'classes_') else REP_CLASSES)
            
            # ใช้ st.metric สำหรับแสดงจำนวนครั้งที่ตรวจจับได้ทั้งหมด
            st.metric(label="จำนวนครั้งที่ตรวจจับได้ทั้งหมด", value=f"{len(rep_results)} ครั้ง")
            st.markdown("---")

            # ใช้ st.expander เพื่อจัดกลุ่ม "การจำแนกประเภทในแต่ละครั้ง"
            with st.expander("รายละเอียดการจำแนกแต่ละรอบ"):
                if rep_results:
                    # สร้าง DataFrame สำหรับแสดงผลแต่ละรอบ
                    rep_df = pd.DataFrame({
                        "รอบที่": [i for i in range(1, len(rep_results) + 1)],
                        "ประเภทท่าทาง": rep_results
                    })
                    st.table(rep_df) # ใช้ st.table เพื่อแสดงผลแบบเรียบง่าย
                else:
                    st.info("ไม่พบการจำแนกประเภทสำหรับแต่ละรอบ.")
            
            st.markdown("---")

            # ใช้ st.expander สำหรับ "สรุปการจำแนกภาพรวม"
            with st.expander("สรุปการจำแนกภาพรวม"):
                if summary:
                    summary_df = pd.DataFrame(summary.items(), columns=["ประเภทท่าทาง", "จำนวนครั้ง"])
                    st.dataframe(summary_df.set_index("ประเภทท่าทาง")) # ใช้ st.dataframe ที่ยืดหยุ่นกว่า
                else:
                    st.info("ไม่พบข้อมูลสรุปการจำแนกภาพรวม.")
        else:
            st.info("ไม่พบการเคลื่อนไหวที่นับเป็นรอบได้ในวิดีโอนี้")
        st.markdown("---")
        # --- สิ้นสุดส่วนแสดงผลลัพธ์โมเดล ---