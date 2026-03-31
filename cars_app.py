import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="توقع أسعار السيارات PRO", page_icon="🏎️")
st.title("🏎️ أداة توقع أسعار السيارات (النسخة الاحترافية)")
st.write("أضفنا ماركة السيارة والممشى لدقة أعلى!")

# --- تحميل النموذج ---
@st.cache_resource 
def load_model():
    return tf.keras.models.load_model("jordan_cars_model_pro.keras")

model = load_model()

# --- قراءة البيانات لاستخراج الماركات تلقائياً ---
df_original = pd.read_csv("cars.csv")
df_original['Brand'] = df_original['Model'].apply(lambda x: str(x).split()[0])
brands_list = sorted(df_original['Brand'].unique().tolist())

# --- واجهة الإدخال الجديدة ---
col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("ماركة السيارة", brands_list)
    year = st.number_input("سنة الصنع", min_value=1990, max_value=2024, value=2015, step=1)

# التعديل هنا: وضعنا "الحالة" و "الممشى" في نفس العمود لنقرأ الحالة أولاً
with col2:
    condition = st.selectbox("حالة السيارة", ["used", "New (Zero)"])
    
    # 💡 التحقق الذكي: إذا كانت الحالة "جديدة"، نجعل الممشى 0 ونقوم بتعطيله
    is_new_car = (condition == "New (Zero)")
    default_mileage = 0 if is_new_car else 75000
    
    mileage = st.number_input(
        "الممشى التقريبي (كم)", 
        min_value=0, 
        max_value=500000, 
        value=default_mileage, 
        step=5000,
        disabled=is_new_car # هذه الخاصية هي التي تمنع المستخدم من التعديل!
    )

with col3:
    transmission = st.selectbox("ناقل الحركة", ["Automatic", "manual"])
    fuel_type = st.selectbox("نوع الوقود", ["gasoline", "Hybrid", "electricity", "diesel"])

# --- التوقع ---
if st.button("توقع السعر 🔮"):
    # تجهيز بيانات المستخدم
    user_data = pd.DataFrame([[year, fuel_type, transmission, condition, brand, mileage]], 
                             columns=['Year', 'Fuel Type', 'Transmission', 'Condition', 'Brand', 'Mileage_Num'])
    
    def clean_mileage_for_app(m):
        m = str(m).replace(' km', '').replace(',', '')
        if '+' in m: return float(m.replace('+', ''))
        elif '-' in m:
            parts = m.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            try: return float(m)
            except: return 0.0

    df_orig_features = df_original[['Year', 'Fuel Type', 'Transmission', 'Condition', 'Brand']].copy()
    df_orig_features['Mileage_Num'] = df_original['Mileage'].apply(clean_mileage_for_app)

    # الدمج والتحويل للأرقام
    df_combined = pd.concat([df_orig_features, user_data], ignore_index=True)
    X_all = pd.get_dummies(df_combined).astype(float).values
    X_user = X_all[-1:]
    
    # التوقع
    prediction = model.predict(X_user)
    predicted_price = prediction[0][0]
    
    if predicted_price > 0:
        st.success(f"💰 السعر المتوقع لسيارة {brand} هو تقريباً: **{predicted_price:,.0f} دينار أردني**")
    else:
        st.error("البيانات المدخلة خارج نطاق التدريب (مثل سيارة قديمة جداً بممشى 0).")
