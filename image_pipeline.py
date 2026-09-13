

import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. قراءة الصور والمعالجة المسبقة واستخراج السمات
# ==========================================
def preprocess_and_extract_features(image_path, target_size=(64, 64)):
    """
    تقرأ الصورة، تجري المعالجة المسبقة، وتستخرج منها متجه السمات (Features).
    """
    # أ. قراءة الصورة
    img = cv2.imread(image_path)
    if img is None:
        return None

    # ب. المعالجة المسبقة (Preprocessing)
    # 1. تغيير الحجم توحيداً للمدخلات
    img_resized = cv2.resize(img, target_size)
    # 2. التحويل لدرجات الرمادي (Grayscale)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # 3. توحيد التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    # 4. تنعيم الصورة وتقليل الضوضاء (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)

    # ج. استخراج السمات (Feature Extraction)
    # 1. سمات الحواف عبر Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 2. سمات المدرج التكراري الألوان/الرمادي (Histogram)
    hist = cv2.calcHist([blurred], [0], None, [32], [0, 256]).flatten()
    
    # دمج السمات في متجه واحد (Feature Vector)
    features = np.hstack([blurred.flatten(), edge_magnitude.flatten(), hist])
    return features

# ==========================================
# 2. تحميل البيانات وتطبيق الأنبوب (Pipeline)
# ==========================================
def load_dataset(dataset_dir):
    X, y = [], []
    classes = sorted(os.listdir(dataset_dir))
    
    print("[*] جاري قراءة الصور وتطبيق المعالجة المسبقة واستخراج السمات...")
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            feat = preprocess_and_extract_features(img_path)
            if feat is not None:
                X.append(feat)
                y.append(label_idx)
                
    return np.array(X), np.array(y), classes

# ==========================================
# 3. التدريب والاختبار وحفظ النموذج
# ==========================================
def main():
    dataset_path = "dataset" # مجلد البيانات يضم مجلدات الفرعيات كفئات
    model_save_path = "trained_image_classifier.pkl"

    # إلغاء التنفيذ إذا لم يكن المجلد موجوداً
    if not os.path.exists(dataset_path):
        print(f"⚠️ يرجى إنشاؤه وتمرير مجلد البيانات في المسار: {dataset_path}")
        return

    # أ. تحميل البيانات والسمات
    X, y, classes = load_dataset(dataset_path)
    print(f"[+] تم استخراج السمات بنجاح. إجمالي العينات: {len(X)}")

    # ب. تقسيم البيانات (80% تدريب - 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ج. تدريب النموذج
    print("[*] جاري تدريب النموذج (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # د. اختبار النموذج وتقييم الأداء
    print("[*] جاري التقييم والاختبار...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] دقة النموذج على بيانات الاختبار: {acc * 100:.2f}%\n")
    print("تقرير التقييم التفصيلي:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # هـ. حفظ النموذج ومسرد الفئات
    pipeline_data = {
        'model': clf,
        'classes': classes
    }
    joblib.dump(pipeline_data, model_save_path)
    print(f"[+] تم حفظ النموذج النهائي بنجاح في: {model_save_path}")

if __name__ == "__main__":
    main()