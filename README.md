# Fake-Face_Detection


In the age of the modern world, machine 
learning and computer vision is playing hard, it is 
going to be the future of technological advances. 
One such advancement is going to be facial 
recognition. Facial recognition is crucial since it 
will replace biometric authentication in the future 
for identity cross-verification. But there is a 
problem with facial recognition. The world is 
making its move towards advancement, it is also 
getting on the worst side to alter the advancement. 
Facial recognition can be altered using open-source 
applications. Altering the face is nothing but 
changing the facial features at pixelated level. A lot 
of work is done in this area but most of them 
worked with SVM classifier with models like 
VGG19, RestNet50, Xception, and LBP (Linear 
Binary Pattern). In this paper we propose to
conduct a comparison study of several CNN model
like RESNET50, LBP(local binary patter),VGG-19 
and Xception with KNN classifier to find out the 
better-performing model and their accuracy. The 
result will be the detection and classification of fake 
and real images from the dataset. The used dataset 
is from the "Real and Fake Face identification" 
deepfake dataset from Yonsei University's 
Computational Intelligence Photography Lab and 
the size is 2100 images which are around 418MB.
Keywords—SVM ,KNN ,CNN ,Fake face Detection, 
Deep Learning.



Fake Face Detection Using Jupyter Notebook

This repository provides a step-by-step implementation of a fake face detection model. The workflow is detailed in the provided Jupyter Notebook file. This guide explains how to set up the environment, execute the notebook, and interpret the results.

Table of Contents

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

Fake face detection is an essential task in the field of AI to differentiate between real and AI-generated faces. This project leverages advanced image analysis techniques to identify such images effectively. The notebook walks through data loading, preprocessing, model training, and evaluation.

Prerequisites

Ensure you have the following installed:

Python 3.8 or above

Jupyter Notebook

Required Python libraries (see requirements.txt or below)

Installation

Clone the repository:

git clone https://github.com/yourusername/fake-face-detection.git
cd fake-face-detection

Install dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook

Dataset

The dataset used in this project includes both real and AI-generated images. Ensure the dataset is downloaded and placed in the data/ directory.

Real faces: [Source link]

AI-generated faces: [Source link]

Notebook Overview

The notebook includes the following sections:

Data Loading: Imports real and fake face datasets for analysis.

Data Visualization: Displays sample images from the dataset for a clear understanding of the input data.

Preprocessing: Prepares the dataset by resizing and normalizing images to ensure compatibility with the model.

Model Training: Implements a Convolutional Neural Network (CNN) to classify images as real or fake.

Evaluation: Tests the trained model on unseen data and calculates performance metrics such as accuracy.

Visualization: Provides visual insights into model performance through accuracy and loss plots.

How to Run

Open the notebook file (fake-face-detection.ipynb) in Jupyter Notebook.

Execute the cells sequentially to:

Load and preprocess the dataset

Train the model

Evaluate and visualize results

Ensure the dataset is correctly structured as expected in the notebook.

Results and Visualization

The model produces the following results:

Accuracy: High accuracy achieved during training and testing phases, demonstrating effective classification of real and fake faces.

Predictions: The model successfully differentiates between real and fake faces, providing a clear prediction for each input image.

Training History: Visual plots illustrate training and validation accuracy trends, along with loss curves, to depict model performance over time.

Example Outputs:

Input image examples are shown with corresponding predictions (e.g., "Real Face" or "Fake Face").

Visual comparisons of correctly and incorrectly classified images are provided for better interpretability.
