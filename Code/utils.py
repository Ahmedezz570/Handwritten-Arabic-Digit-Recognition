import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_and_save_learning_curves(history):


    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))
    plt.close()


    #