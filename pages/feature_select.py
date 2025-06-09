import streamlit as st
import os # Import os for path manipulation

def app():
    st.title("⚙️ เลือกการประมวลผล")
    st.markdown("---")

    st.subheader("1. วิดีโอของคุณ")
    video_source_option = st.radio(
        "อัปโหลดวิดีโอการทำกายภาพของคุณ:",
        ("อัปโหลดไฟล์วิดีโอ"), # เพิ่ม "ใช้กล้องเว็บแคม" กลับเข้ามา
        key="video_source_radio"
    )
    
    st.session_state.video_source = video_source_option

    # Conditional display based on video source selection
    if st.session_state.video_source == "อัปโหลดไฟล์วิดีโอ":
        uploaded_file = st.file_uploader("เลือกไฟล์วิดีโอ (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            # Save the uploaded file object to session state for processing.py to use
            st.session_state.uploaded_video_file = uploaded_file
            # Store the name of the uploaded file
            st.session_state.uploaded_video_name = uploaded_file.name
            st.success(f"อัปโหลดไฟล์ '{uploaded_file.name}' เรียบร้อยแล้ว")
        else:
            st.session_state.uploaded_video_file = None
            st.session_state.uploaded_video_name = None # Clear name if no file is uploaded
            st.info("ยังไม่มีไฟล์วิดีโออัปโหลด")

    st.markdown("---")

    st.subheader("2. เลือกแขนข้างที่ต้องการประเมิน")
    selected_arm_option = st.radio(
        "โปรดระบุแขนข้างที่ทำกายภาพบำบัด:",
        ("ซ้าย (Left Arm)", "ขวา (Right Arm)"), # สามารถเพิ่ม "ทั้งสองข้าง (Both Arms)" ได้ถ้ามีโมเดลรองรับ
        key="selected_arm_radio"
    )
    st.session_state.selected_arm = selected_arm_option

    st.markdown("---")

    st.subheader("3. เลือกโมเดล AI")
    st.write("เลือกโมเดล AI ที่ต้องการใช้ในการวิเคราะห์ท่าทาง:")

    # Define model options and their paths (replace with your actual paths)
    model_options = {
        "โมเดล 33 จุด (Raw MediaPipe Landmarks)": "models\movemate_raw_data_rf.pkl", # ตัวอย่าง
        "โมเดล 17 จุด (Selected MediaPipe Landmarks)": "models\movemate_17keys_rf.pkl", # ตัวอย่าง
        "โมเดล 17 จุด + Features (ระยะห่าง/มุม)": "models\movemate_feature_rf.pkl" # ตัวอย่าง
    }

    selected_model_display = st.selectbox(
        "เลือกประเภทโมเดล AI:",
        list(model_options.keys()),
        key="selected_model_selectbox"
    )
    st.session_state.selected_model_type = selected_model_display
    st.session_state.model_path = model_options[selected_model_display]
    
    st.markdown("---")

    # The button to proceed
    if st.button("พร้อมแล้ว! ไปหน้าประมวลผล"):
        # Basic validation before proceeding
        if st.session_state.video_source == "อัปโหลดไฟล์วิดีโอ" and st.session_state.uploaded_video_file is None:
            st.warning("กรุณาอัปโหลดไฟล์วิดีโอให้เรียบร้อยก่อนดำเนินการต่อ")
        elif st.session_state.model_path is None:
            st.warning("กรุณาเลือกโมเดล AI")
        else:
            st.success("กรุณาเลือก **'3. Processing Results'** ที่แถบเมนูด้านซ้ายเพื่อไปขั้นตอนถัดไป")
