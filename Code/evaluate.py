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
    # Removed Arabic labels
    metrics = ['Test Loss', 'Test Accuracy']
    values = [loss, accuracy]

    plt.figure(figsize=(7, 5))
    # Create Bar Plot
    bars = plt.bar(metrics, values, color=['#FF6347', '#3CB371'])  # Different colors

    plt.title('Final Performance Evaluation')
    plt.ylim(0.0, 1.0)  # Set range from 0 to 1 for loss and accuracy
    plt.ylabel('Value')

    # Add value of each bar on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

    plt.show()


def evaluate_model():
    # Load data for evaluation (Test Set)
    _, _, X_test, _, _, y_test_cat, y_test_raw = load_and_prepare_data()

    # Check for model existence
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at path: {MODEL_PATH}")
        print("Please ensure train.py was run first.")
        return

    # Load the best model
    model = load_model(MODEL_PATH)

    # 1. Overall Performance Evaluation
    print("\n--- Overall Performance Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot accuracy and loss
    plot_evaluation_metrics(loss, accuracy)

    # 2. Make Predictions
    print("\nMaking predictions on the test set...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # 3. Classification Report
    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test_raw,
            y_pred_classes,
            target_names=TARGET_NAMES
        )
    )

    # 4. Confusion Matrix Plot
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
    plt.title("Confusion Matrix")  # Removed Arabic
    plt.ylabel("True Label")  # Removed Arabic
    plt.xlabel("Predicted Label")  # Removed Arabic
    plt.show()


if __name__ == "__main__":
    evaluate_model()