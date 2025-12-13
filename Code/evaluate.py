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


def plot_evaluation_metrics(loss, accuracy):

    metrics = ['Test Loss', 'Test Accuracy']
    values = [loss, accuracy]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(metrics, values, color=['#FF6347', '#3CB371'])

    plt.title('Final Performance Evaluation')
    plt.ylim(0.0, 1.0)
    plt.ylabel('Value')


    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

    plt.show()


def evaluate_model():

    _, _, X_test, _, _, y_test_cat, y_test_raw = load_and_prepare_data()


    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at path: {MODEL_PATH}")
        print("Please ensure train.py was run first.")
        return


    model = load_model(MODEL_PATH)


    print("\n--- Overall Performance Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


    plot_evaluation_metrics(loss, accuracy)


    print("\nMaking predictions on the test set...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)


    print("\n--- Classification Report ---")
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
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


if __name__ == "__main__":
    evaluate_model()