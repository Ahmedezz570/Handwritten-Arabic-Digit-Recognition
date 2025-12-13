# Project: Arabic Handwritten Digit Recognition using CNN (Project_2_12)

##  1. Project Description

This project implements a Convolutional Neural Network (CNN) to achieve high-accuracy classification of handwritten Arabic digits (0-9). The model utilizes advanced techniques like **Batch Normalization** and **Adam Optimizer** with a learning rate scheduler to achieve state-of-the-art performance.

## 2. Dataset

* **Dataset Name:** Extended Arabic Handwritten Digits Dataset (AHDD-like distribution).
* **Total Images:** 70,000 images (60,000 for training/validation, 10,000 for testing).
* **Image Size:** $28 \times 28 \times 1$ (Grayscale).

## **Dataset Link:**
## [Insert the actual Kaggle or source link here, e.g., https://www.kaggle.com/datasets/mloey1/ahdd1/data]

## 🛠 3. How to Install Dependencies

The project requires Python 3.x and the following libraries. It is recommended to install them in a virtual environment.

```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn
```
## How to Run the Training Script
The training script (train.py) loads and preprocesses the data, builds the CNN model, and initiates training using the defined hyperparameters and callbacks (Early Stopping, Model Checkpoint, ReduceLROnPlateau).

Instructions: Ensure all four CSV data files are placed inside the data/ directory.

```
python code/train.py
```
### Outputs:

Saves the best model weights to results/best_model.h5.

## 5. How to Run Evaluation
The evaluation script (evaluate.py) loads the best_model.h5 and assesses its performance on the independent test set (10,000 images).
```
python code/evaluate.py
```
### Outputs:

Prints the Classification Report (Precision, Recall, F1-Score).

Displays the Confusion Matrix.

Displays the overall Test Accuracy and Test Loss.

## How to Load the Saved Model for Inference
To use the trained model for making predictions on new data (inference), you can load the saved H5 file:
```
import os
from tensorflow.keras.models import load_model

# Define the path to the best model
MODEL_PATH = os.path.join("results", "best_model.h5")

# Load the model
trained_model = load_model(MODEL_PATH)

# Example: Predict on a single new image (new_image_processed)
# prediction = trained_model.predict(new_image_processed)
```
