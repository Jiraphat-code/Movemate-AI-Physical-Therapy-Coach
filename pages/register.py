import streamlit as st
'''
def app():
    st.title("MoveMate - Your AI Rehab Companion at home")
    st.markdown("""
    ## ยินดีต้อนรับสู่ MoveMate
    ระบบผู้ช่วยกายภาพบำบัดด้วย AI สำหรับนักกายภาพและผู้ป่วย
    กรุณากรอกข้อมูลของคุณเพื่อเริ่มต้นใช้งาน
    """)

    # ฟอร์มกรอกข้อมูล
    with st.form(key="user_info_form"):
        first_name = st.text_input("ชื่อ", value=st.session_state.get("first_name", ""))
        last_name = st.text_input("สกุล", value=st.session_state.get("last_name", ""))
        age = st.number_input("อายุ", min_value=1, max_value=120, value=st.session_state.get("age", 18))
        purpose = st.text_area("จุดประสงค์ที่ทำ rehab", value=st.session_state.get("purpose", ""))
        submitted = st.form_submit_button("ถัดไป (Next)")

    # เก็บข้อมูลใน session_state
    if submitted:
        st.session_state.first_name = first_name
        st.session_state.last_name = last_name
        st.session_state.age = age
        st.session_state.purpose = purpose
        
        # 1. แสดงสถานะโดยรวมแบบกระชับใน st.success()
        st.success("บันทึกข้อมูลสำเร็จ! ✨")
        
        # 2. แสดงรายละเอียดข้อมูลผู้ใช้แยกต่างหากด้วย st.write() หรือ st.markdown()
        #    ซึ่งรองรับการขึ้นบรรทัดใหม่
        st.write(f"**ข้อมูลผู้ใช้งาน:**")
        st.write(f"- ชื่อ-สกุล: {st.session_state.first_name} {st.session_state.last_name}")
        st.write(f"- อายุ: {st.session_state.age} ปี")
        st.write(f"- จุดประสงค์: {st.session_state.purpose}")
        
        # 3. ให้คำแนะนำสำหรับการดำเนินการถัดไป
        st.info("กรุณาเลือก **'2. Feature Selection'** ที่แถบเมนูด้านซ้ายเพื่อไปขั้นตอนถัดไป")
'''

import streamlit as st

def app():
    st.title("📝 ฟอร์มลงทะเบียนผู้ป่วย")
    st.markdown("---")

    with st.form("patient_registration_form"):
        st.subheader("ข้อมูลพื้นฐาน")
        patient_id = st.text_input("รหัสผู้ป่วย", key="reg_patient_id")
        full_name = st.text_input("ชื่อ-นามสกุล", key="reg_full_name")
        age = st.number_input("อายุ", min_value=0, max_value=120, key="reg_age")
        gender = st.selectbox("เพศ", ["ชาย", "หญิง"], key="reg_gender")

        st.subheader("ข้อมูลการติดต่อ")
        phone = st.text_input("เบอร์โทรศัพท์", key="reg_phone")

        st.subheader("ข้อมูลสุขภาพเบื้องต้น")
        diagnosis = st.text_area("การวินิจฉัย/อาการเบื้องต้น", key="reg_diagnosis")
        
        submitted = st.form_submit_button("บันทึกข้อมูล")

        if submitted:
            if patient_id and full_name:
                st.session_state.patient_info = {
                    "รหัสคนไข้": patient_id,
                    "ชื่อ-นามสกุล": full_name,
                    "อายุ": age,
                    "เพศ": gender,
                    "เบอร์โทรศัพท์": phone,
                    "การวินิจฉัย/อาการ": diagnosis
                }
                st.success("บันทึกข้อมูลคนไข้เรียบร้อยแล้ว!")
            else:
                st.warning("กรุณากรอกรหัสคนไข้และชื่อ-นามสกุล")