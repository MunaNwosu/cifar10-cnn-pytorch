# CIFAR-10 Image Classification using PyTorch

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The model is trained using data augmentation techniques to improve generalization and achieve higher accuracy.

## Features

Uses a CNN architecture for better image classification

Implements data augmentation (random cropping & horizontal flipping) to improve generalization

Uses CrossEntropyLoss for multi-class classification

Trained using the Adam optimizer with a learning rate of 0.001

Evaluates model performance on the CIFAR-10 test set

## Dataset

The CIFAR-10 dataset consists of 60,000 color images (32x32 pixels) in 10 classes, with 6,000 images per class. Classes include:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Model Architecture

The implemented CNN consists of:

Two convolutional layers with ReLU activation and max-pooling

Fully connected layers to output 10 class predictions

Softmax is not applied in the final layer (since CrossEntropyLoss handles it)

## Installation & Usage

### Requirements

Ensure you have Python 3.7+ and install the dependencies:

pip install torch torchvision

Training the Model

### Run the script to train the model:

python train.py

Testing the Model

After training, the model evaluates on the test dataset
