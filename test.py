import cv2
import joblib
import numpy as np
from image_pipeline import preprocess_and_extract_features

def predict_single_image(image_path, model_path="trained_image_classifier.pkl"):
    # 1. تحميل النموذج والفئات المحفوظة
    saved_data = joblib.load(model_path)
    model = saved_data['model']
    classes = saved_data['classes']

    # 2. قراءة ومعالجة واستخراج سمات الصورة الجديدة
    features = preprocess_and_extract_features(image_path)
    if features is None:
        print("❌ فشل في قراءة الصورة.")
        return

    # 3. التنبؤ بالفئة
    features = np.expand_dims(features, axis=0) # تحويل إلى batch لعيّنة واحدة
    prediction_idx = model.predict(features)[0]
    predicted_class = classes[prediction_idx]

    print(f"📷 الصورة: {image_path}")
    print(f"🎯 التوقع: {predicted_class}")

if __name__ == "__main__":
    # استبدل المسار بصورتك للاختبار
    predict_single_image("/home/abdullah/Desktop/imp/test/dogs/dog.4001.jpg")