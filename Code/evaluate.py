import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import load_and_prepare_data


MODEL_DIR = "results"

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")


TARGET_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def evaluate_model():

    print("جاري تحميل ومعالجة بيانات الاختبار...")
    _, _, X_test, _, _, y_test_cat, y_test_raw = load_and_prepare_data()


    if not os.path.exists(MODEL_PATH):
        print(f"خطأ: النموذج غير موجود في المسار: {MODEL_PATH}")
        print("الرجاء التأكد من تشغيل train.py أولاً.")
        return

    print(f"جاري تحميل النموذج من: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print("\n--- تقييم الأداء العام ---")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"خسارة الاختبار (Test Loss): {loss:.4f}")
    print(f"دقة الاختبار (Test Accuracy): {accuracy:.4f}")

    print("\nجاري إجراء التنبؤات على مجموعة الاختبار...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print("\n--- تقرير التصنيف (Classification Report) ---")
    print(
        classification_report(
            y_test_raw,
            y_pred_classes,
            target_names=TARGET_NAMES
        )
    )

    cm = confusion_matrix(y_test_raw, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES
    )
    plt.title(" (Confusion Matrix)")
    plt.ylabel(" (True Label)")
    plt.xlabel("(Predicted Label)")
    plt.show()


if __name__ == "__main__":
    evaluate_model()