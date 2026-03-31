import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os

# --- إعدادات الصفحة ---
st.set_page_config(page_title="توقع أسعار السيارات في الأردن", page_icon="🚗")
st.title("🚗 أداة الذكاء الاصطناعي لتوقع أسعار السيارات")
st.write("أدخل مواصفات السيارة أدناه لمعرفة السعر المتوقع في السوق الأردني.")

# --- تحميل النموذج والبيانات المرجعية ---
@st.cache_resource # لتسريع التطبيق وعدم تحميل النموذج مع كل ضغطة
def load_model():
    return tf.keras.models.load_model("jordan_cars_model_improved.keras")

model = load_model()

# --- واجهة الإدخال ---
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("سنة الصنع", min_value=1990, max_value=2024, value=2015, step=1)
    transmission = st.selectbox("ناقل الحركة (الجير)", ["Automatic", "manual"])

with col2:
    fuel_type = st.selectbox("نوع الوقود", ["gasoline", "Hybrid", "electricity", "diesel"])
    condition = st.selectbox("حالة السيارة", ["used", "New (Zero)"])

# --- زر التوقع ---
if st.button("توقع السعر 🔮"):
    # 1. إنشاء جدول ببيانات المستخدم
    user_data = pd.DataFrame([[year, fuel_type, transmission, condition]], 
                             columns=['Year', 'Fuel Type', 'Transmission', 'Condition'])
    
    # 2. قراءة بيانات السيارات الأصلية لمعرفة الأعمدة المطلوبة للتحويل (One-Hot Encoding)
    # لا تقلق، هذا فقط لضبط شكل البيانات ولا يؤثر على التوقع
    df_original = pd.read_csv("cars.csv")[['Year', 'Fuel Type', 'Transmission', 'Condition']]
    
    # دمج بيانات المستخدم مع البيانات الأصلية
    df_combined = pd.concat([df_original, user_data], ignore_index=True)
    
    # تحويل النصوص إلى أرقام (Dummy variables)
    X_all = pd.get_dummies(df_combined).astype(float).values
    
    # استخراج الصف الأخير فقط (وهو صف بيانات المستخدم بعد تحويله لأرقام)
    X_user = X_all[-1:]
    
    # 3. توقع السعر باستخدام النموذج
    prediction = model.predict(X_user)
    predicted_price = prediction[0][0]
    
    # 4. عرض النتيجة
    if predicted_price > 0:
        st.success(f"💰 السعر المتوقع للسيارة هو تقريباً: **{predicted_price:,.0f} دينار أردني**")
    else:
        st.error("البيانات المدخلة غير منطقية أو خارج نطاق تدريب النموذج.")