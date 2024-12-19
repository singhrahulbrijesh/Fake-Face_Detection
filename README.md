## Fake Face Detection
Overview
In today's world, machine learning and computer vision are becoming integral parts of technological advancement. Facial recognition, in particular, is poised to replace biometric authentication for identity verification. However, facial recognition systems are vulnerable to manipulation using open-source tools that can alter facial features at the pixel level.

This project conducts a comparison study of various Convolutional Neural Network (CNN) models, including ResNet50, VGG19, Xception, and Local Binary Pattern (LBP), combined with classifiers like KNN, to determine the most effective method for detecting fake faces. The study uses the "Real and Fake Face Identification" deepfake dataset from Yonsei University's Computational Intelligence Photography Lab.

Keywords: KNN, CNN, Fake Face Detection, Deep Learning

## Table of Contents
Introduction
Prerequisites
Installation
Dataset
Notebook Overview
How to Run
Results and Visualization
Contributing
License
Introduction
Fake face detection is an essential task in the field of AI to differentiate between real and AI-generated faces. This project leverages advanced image analysis techniques to identify such images effectively. The provided Jupyter Notebook guides users through data loading, preprocessing, model training, and evaluation.

## Prerequisites
Ensure you have the following installed:

## Python 3.8 or above
Jupyter Notebook
Required Python libraries (listed in requirements.txt)
Installation

## Clone the repository:
git clone -> https://github.com/singhrahulbrijesh/fake-face-detection.git

cd fake-face-detection

## Install dependencies:
pip install -r requirements.txt
Launch Jupyter Notebook:


jupyter notebook
Dataset
The dataset contains real and AI-generated images. Ensure it is downloaded and placed in the data/ directory.

Real Faces: Source Link
AI-Generated Faces: Source Link
Notebook Overview
The notebook consists of the following sections:

## Data Loading: Imports and loads real and fake face datasets for analysis.
Data Visualization: Displays sample images to provide a better understanding of the dataset.
Preprocessing: Resizes and normalizes images for compatibility with the model.
Model Training: Implements CNN models to classify images as real or fake.
Evaluation: Tests the trained model on unseen data and computes performance metrics like accuracy.
Visualization: Plots training history and highlights predictions to illustrate the model's effectiveness.

How to Run
Open the fake-face-detection.ipynb notebook in Jupyter Notebook.
Follow these steps:
Load and preprocess the dataset.
Train the model.
Evaluate the results and generate visualizations.
Ensure the dataset structure matches the format expected in the notebook.
Results and Visualization
The project yields the following results:

## Accuracy: High training and testing accuracy, confirming effective classification of real and fake faces.
Predictions: The model correctly predicts "Real Face" or "Fake Face" for input images.
Training History: Plots of training and validation accuracy and loss provide insights into the model's learning process.

## Example Outputs:
Input Image: Example real and fake faces are visualized alongside their predictions.

## Real Faces
![Real Face](https://github.com/user-attachments/assets/80f829f1-aa4a-4ada-ac55-dc2a344b6821)

## Fake Faces
![Fake Faces](https://github.com/user-attachments/assets/134fa5c2-01be-4116-ba16-706267d1d059)

## Correctly and Incorrectly Classified Images: Visual comparison of accurate and erroneous classifications highlights areas for improvement.

## Model Accuracy
Based on the notebook's outputs:


### Model Accuracy Comparison

| **Model**                      | **Training Accuracy** | **Testing Accuracy** | **Notes**                                           |
|--------------------------------|-----------------------|-----------------------|----------------------------------------------------|
| **ResNet50 (Pretrained)**      | 95.16%               | 94.36%               | Pretraining on ImageNet boosts performance.       |
| **ResNet50 (Not Pretrained)**  | 84.37%               | 83.97%               | Lower accuracy due to lack of pretraining.        |
| **VGG19**                      | 92.33%               | 92.46%               | Consistently high accuracy across training/testing.|
| **Xception**                   | 96.33%               | 95.17%               | Best performing model in both training and testing.|
| **Local Binary Pattern (LBP)** | 80.22%               | 78.81%               | Simpler feature-based method, less effective.     |
| **KNN (with CNN features)**    | 76.32%               | 74.65%               | Performance depends on extracted features.        |


## Contributing
Contributions are welcome! If you have ideas or find issues, please open an issue or submit a pull request.

