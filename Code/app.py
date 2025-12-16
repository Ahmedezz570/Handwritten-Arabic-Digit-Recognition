import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time
import os


IMG_ROWS, IMG_COLS = 28, 28
DIGITS_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MODEL_PATH = "results/best_model.h5"


try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    MODEL = load_model(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Failed to load model. Please ensure {MODEL_PATH} exists.")



def predict_digit(image: Image.Image):
    if image is None:
        return 0, 0.0, 0.0
    start_time = time.time()
    img_gray = image.convert('L')
    img_resized = img_gray.resize((IMG_ROWS, IMG_COLS))
    X = np.array(img_resized).flatten()
    X = X.astype('float32')
    X = X / 255.0
    X = X.reshape(1, IMG_ROWS, IMG_COLS, 1)
    predictions = MODEL.predict(X, verbose=0)
    end_time = time.time()
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_digit = DIGITS_MAP[predicted_index]
    confidence = np.max(predictions)
    prediction_time = end_time - start_time
    return predicted_digit, prediction_time, confidence



image_input = gr.Image(type="pil", label="", height=200)
output_digit = gr.Number(label="Predicted Digit", precision=0)
output_time = gr.Number(label="Prediction Time (s)", precision=4)

if 'MODEL' in locals() and MODEL is not None:
    gr.Interface(
        fn=lambda img: (
            lambda d, t, c: (d, t, c * 100)
        )(*predict_digit(img)),
        inputs=image_input,
        outputs=[output_digit, output_time],
        title="Digit Recognition Tester",
        description="",
    ).launch()
else:
    print("Gradio interface not launched due to model loading error.")