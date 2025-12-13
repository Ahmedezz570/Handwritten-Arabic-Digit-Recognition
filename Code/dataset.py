import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
# إضافة مكتبة kagglehub
import kagglehub

IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NUM_CLASSES = 10

# هنا نقوم بتعريف أسماء الملفات فقط، وليس المسارات الثابتة
TRAIN_IMAGES_NAME = 'csvTrainImages 60k x 784.csv'
TRAIN_LABEL_NAME = 'csvTrainLabel 60k x 1.csv'
TEST_IMAGES_NAME = 'csvTestImages 10k x 784.csv'
TEST_LABEL_NAME = 'csvTestLabel 10k x 1.csv'


def preprocess_data(X):
    # ... (الدالة تبقى كما هي)
    X = X.astype('float32') / 255.0
    X = X.reshape(X.shape[0], IMG_ROWS, IMG_COLS, 1)
    return X


def load_and_prepare_data(train_size=0.9):
    # 1. تحميل مجموعة البيانات وتحديد المسار
    # سيقوم هذا السطر بتنزيل الملفات وإرجاع مسار المجلد الذي يحتوي عليها
    DATA_PATH = kagglehub.dataset_download("mloey1/ahdd1")
    print(f"تم تحميل البيانات إلى المسار: {DATA_PATH}")

    # 2. تحديد المسارات الكاملة للملفات
    TRAIN_FILE_PATH = os.path.join(DATA_PATH, TRAIN_IMAGES_NAME)
    TRAIN_LABEL_PATH = os.path.join(DATA_PATH, TRAIN_LABEL_NAME)
    TEST_FILE_PATH = os.path.join(DATA_PATH, TEST_IMAGES_NAME)
    TEST_LABEL_PATH = os.path.join(DATA_PATH, TEST_LABEL_NAME)

    # 3. قراءة الملفات باستخدام المسارات الجديدة
    X_train_raw = pd.read_csv(TRAIN_FILE_PATH, header=None).values
    y_train_raw = pd.read_csv(TRAIN_LABEL_PATH, header=None).values.flatten()

    X_test_raw = pd.read_csv(TEST_FILE_PATH, header=None).values
    y_test_raw = pd.read_csv(TEST_LABEL_PATH, header=None).values.flatten()

    # ... (بقية منطق تقسيم البيانات والتشكيل يبقى كما هو)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=(1 - train_size), random_state=42, stratify=y_train_raw
    )

    X_train_proc = preprocess_data(X_train)
    X_val_proc = preprocess_data(X_val)
    X_test_proc = preprocess_data(X_test_raw)

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test_raw, NUM_CLASSES)

    return X_train_proc, X_val_proc, X_test_proc, y_train_cat, y_val_cat, y_test_cat, y_test_raw