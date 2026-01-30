**Handwritten Digit Classification using Neural Networks (TensorFlow/Keras)**

📌 Project Overview

This project focuses on classifying handwritten digits (0–9) using a Deep Neural Network built with TensorFlow and Keras.
The model is trained on the popular MNIST dataset, which contains grayscale images of handwritten digits.

The goal of this project is to understand how neural networks work for image classification and how model performance is evaluated.


📊 Dataset

Dataset: MNIST Handwritten Digits

Training samples: 60,000 images

Testing samples: 10,000 images

Image size: 28 × 28 pixels (grayscale)

Each image is flattened into a vector of 784 features before being fed into the neural network.


🧠 Model Architecture

The neural network is built using keras.Sequential and consists of:

Input layer of size 784

One hidden Dense layer

Output Dense layer with 10 neurons (one for each digit 0–9)

Activation functions and optimizer are chosen to efficiently learn digit patterns.


⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn


📈 Training & Evaluation

The model is trained for multiple epochs on the training dataset.

Accuracy is used as the primary performance metric.

A confusion matrix is plotted to analyze prediction performance across all digit classes.


📌 Results

Achieved high training accuracy on handwritten digit classification.

The confusion matrix shows how well the model predicts each digit and where misclassifications occur.


🎯 Key Learnings

Understanding how neural networks process image data

Importance of flattening image inputs

Role of loss functions, optimizers, and epochs

Evaluating models using accuracy and confusion matrices

