import tensorflow as tf
import pandas as pd
import numpy as np
import os

print("جاري تجهيز البيانات...")

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_directory, "cars.csv")
df = pd.read_csv(csv_path)

# تنظيف البيانات
df = df.dropna(subset=['Price'])
df['Price'] = df['Price'].str.replace(' JOD', '', regex=False)
df['Price'] = df['Price'].str.replace(',', '', regex=False)
df['Price'] = df['Price'].astype(float)

y = df['Price'].values 

# اختيار المدخلات
features = ['Year', 'Fuel Type', 'Transmission', 'Condition']
df_features = df[features]
X = pd.get_dummies(df_features).astype(float).values

# بناء النموذج
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X)

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(128, activation="relu"), # زدنا عدد الخلايا لزيادة ذكاء النموذج
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# هنا التغيير المهم: استخدمنا MAE بدلاً من MSE
# MAE = Mean Absolute Error (متوسط الخطأ المطلق)
model.compile(
    loss='mae', # سيعطيك الخطأ بالدينار مباشرة
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005) # تقليل سرعة التعلم لدقة أعلى
)

print("جاري التدريب (قد يستغرق بعض الوقت)...")
# زدنا عدد جولات التدريب (epochs) إلى 200
history = model.fit(X, y, epochs=200, verbose=1, validation_split=0.2) 

# حفظ النموذج
model_path = os.path.join(current_directory, "jordan_cars_model_improved.keras")
model.save(model_path)

print("✅ انتهى التدريب بنجاح!")

# طباعة الخطأ النهائي بشكل مفهوم
final_loss = history.history['loss'][-1]
print(f"📉 متوسط نسبة الخطأ في توقع السعر: ± {final_loss:,.0f} دينار تقريباً")