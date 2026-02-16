import streamlit as st
import pandas as pd
import pickle

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Riding Mower Prediction", layout="centered")

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_trained_model():
    with open('RidingMower_Model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_trained_model()

# ส่วนของ UI
st.title("🚜 Riding Mower Ownership Predictor")
st.markdown("กรุณาระบุข้อมูลเพื่อทำนายความเป็นเจ้าของเครื่องตัดหญ้า")

# สร้างฟอร์มรับค่า
with st.sidebar:
    st.header("ใส่ค่าตัวแปร")
    income = st.number_input("รายได้ (Income) ต่อปี ($000)", min_value=0.0, value=60.0)
    lot_size = st.number_input("ขนาดที่ดิน (Lot Size) (000 sq.ft)", min_value=0.0, value=18.0)

# สร้างปุ่มกดทำนาย
if st.button("ทำนายผล"):
    # เตรียมข้อมูลสำหรับทำนาย
    input_data = pd.DataFrame([[income, lot_size]], columns=['Income', 'Lot_Size'])
    
    # ทำนายผล
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.subheader("ผลลัพธ์:")
    if prediction[0] == 1:
        st.success("✅ ทำนายว่า: เป็นเจ้าของ (Owner)")
    else:
        st.warning("❌ ทำนายว่า: ไม่ได้เป็นเจ้าของ (Nonowner)")
    
    # แสดงค่าความมั่นใจ
    st.write(f"ความน่าจะเป็นที่่จะเป็นเจ้าของ: {prob[0][1]:.2%}")