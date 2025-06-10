import streamlit as st
import os # Import os for path manipulation

def app():

    st.header("📚 ท่ากายภาพบำบัด: Shoulder Flexion")
    st.write(
        """
        ท่า **Shoulder Flexion** เป็นท่ากายภาพที่สำคัญในการฟื้นฟูการทำงานของกล้ามเนื้อในช่วงไหล่และแขน
        ในผู้ป่วยหลังภาวะ stroke หรือการบาดเจ็บอื่นๆ โดยเป็นท่าเริ่มต้นสำหรับผู้ป่วยในการฝึกเพื่อฟื้นฟูร่างกาย
        """
    )

    st.subheader("🚀 เป้าหมายของท่านี้:")
    st.write(
        """
        เป้าหมายหลักคือการยกแขนขึ้นไป เหนือศีรษะให้ได้มากที่สุด จนแขนของคุณอยู่ในแนวเดียวกับลำตัว (หรือใกล้เคียง) 
        ซึ่งหมายถึงการได้พิสัยการเคลื่อนไหว (Range of Motion) ที่สมบูรณ์
        """
    )

    st.subheader("📝 ขั้นตอนการทำท่าที่ถูกต้อง (ในท่านอน):")
    st.markdown(
        """
        1.  **จัดท่าเริ่มต้น:**
            * **นอนหงายราบไปกับพื้นหรือเสื่อออกกำลังกาย:** ให้ศีรษะและหลังแนบสนิทกับพื้นตลอดเวลา **ห้ามแอ่นหลัง** โดยเด็ดขาด เพราะจะทำให้กล้ามเนื้อหลังทำงานชดเชยแทนการทำงานของหัวไหล่
            * **ตำแหน่งแขน:** แขนข้างที่จะฝึกเหยียดตรง วางแนบข้างลำตัวโดยให้ **ฝ่ามือหงายขึ้น (Supination)** หรือ **นิ้วหัวแม่มือชี้ขึ้นด้านบน**
            * **ควบคุมลำตัว:** อีกแขนหนึ่งวางข้างลำตัวสบายๆ หรือวางบนหน้าท้องเพื่อช่วยให้ลำตัวมั่นคง

        2.  **การเคลื่อนไหว (Flexion Phase):**
            * **ยกแขนขึ้นช้าๆ:** ค่อยๆ ยกแขนที่เหยียดตรงขึ้นไปด้านหน้าเหนือศีรษะ โดยพยายามรักษา **แขนให้ตรงตลอดเวลา**
            * **โฟกัสที่การทำงานของหัวไหล่:** รู้สึกถึงการทำงานของกล้ามเนื้อหัวไหล่และด้านหน้าของลำตัวขณะยกแขน
            * **ระวังอย่าให้ไหล่ยักขึ้น:** พยายามกดหัวไหล่ให้ห่างจากหู และ **อย่าให้แผ่นหลังยกขึ้นจากพื้น** ในขณะยกแขน
            * **ยกให้สุดพิสัย:** พยายามยกแขนให้สูงที่สุดเท่าที่จะทำได้ จนกระทั่งแขนของคุณอยู่ **เหนือศีรษะและชี้ไปด้านหลัง** หากทำได้ถึงขั้นนี้ แสดงว่าคุณได้พิสัยการเคลื่อนไหวที่เกือบสมบูรณ์แล้ว

        3.  **การควบคุม (Lowering Phase):**
            * **ลดแขนลงช้าๆ และควบคุม:** ค่อยๆ ลดแขนลงกลับสู่ท่าเริ่มต้นอย่างช้าๆ ในแนวทางเดิม
            * **ห้ามปล่อยแขนทิ้ง:** การปล่อยแขนทิ้งลงมาอย่างรวดเร็วอาจทำให้ข้อไหล่บาดเจ็บได้
            * **รักษาฟอร์ม:** ตรวจสอบให้แน่ใจว่าหลังยังคงแนบพื้นและหัวไหล่ไม่ยักขึ้นขณะลดแขนลง

        4.  **การหายใจ:**
            * **หายใจเข้า:** เมื่อยกแขนขึ้น
            * **หายใจออก:** เมื่อลดแขนลง
        """
    )

    st.subheader("⚠️ ข้อควรระวังและความถูกต้องของฟอร์ม:")
    st.markdown(
        """
        * **หลังต้องแนบพื้นเสมอ:** นี่คือสิ่งสำคัญที่สุดในการป้องกันการบาดเจ็บและเพื่อให้กล้ามเนื้อไหล่ทำงานได้เต็มที่ หากหลังแอ่น แสดงว่าคุณใช้กล้ามเนื้อหลังมาช่วยยกแขนแทนกล้ามเนื้อไหล่
        * **แขนต้องตรง:** พยายามรักษาแขนให้เหยียดตรงตลอดการเคลื่อนไหว หากงอข้อศอก แสดงว่าคุณกำลังใช้กล้ามเนื้อต้นแขนช่วยมากเกินไป
        * **หัวไหล่ไม่ยักขึ้น:** ระวังอย่าให้หัวไหล่ของคุณยกขึ้นไปชิดหูในขณะยกแขน เพราะจะทำให้กล้ามเนื้อบ่าทำงานมากเกินไปและลดประสิทธิภาพของท่า
        * **ควบคุมการเคลื่อนไหว:** ทำท่าช้าๆ และควบคุมทุกช่วงการเคลื่อนไหว ทั้งขึ้นและลง เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่และลดความเสี่ยงของการบาดเจ็บ
        """)

    st.subheader("▶️ วีดีโอสาธิตท่า Shoulder Flexion:")
    # ตรวจสอบ URL วีดีโอของคุณตรงนี้
    # ถ้าเป็น YouTube:
    st.video("https://www.youtube.com/watch?v=6iEEWUjOkC0") 

    st.write(
        """
        ดูวีดีโอสาธิตเพื่อทำความเข้าใจท่าทางที่ถูกต้อง และเตรียมพร้อมสำหรับการฝึกของคุณ
        """
    )
    
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
        "โมเดล 33 จุด (Raw MediaPipe Landmarks)": "models/movemate_raw_data_no_tremor_rf.pkl", 
        "โมเดล 17 จุด (Selected MediaPipe Landmarks)": "models/movemate_17keys_rf.pkl", 
        "โมเดล 17 จุด + Features (ระยะห่าง/มุม)": "models/movemate_feature_rf.pkl" 
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