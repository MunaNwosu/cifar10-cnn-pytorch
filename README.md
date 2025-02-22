# Image Classification using PyTorch

## Overview

This project implements two different neural network models using PyTorch to classify images from the MNIST and CIFAR-10 datasets.

MNIST: A fully connected neural network (FCNN) to classify handwritten digits (0-9).

CIFAR-10: A convolutional neural network (CNN) to classify objects into 10 categories.

## Features

Supports both MNIST and CIFAR-10 datasets

Uses fully connected layers for MNIST and CNN layers for CIFAR-10

Implements dropout and data augmentation to improve generalization

Uses CrossEntropyLoss for multi-class classification

Trained using the Adam optimizer with a learning rate of 0.001

## Dataset Details

### MNIST

Consists of 60,000 grayscale images (28x28 pixels) in 10 classes (digits 0-9)

Trained using a fully connected neural network (FCNN)


### CIFAR-10

Consists of 60,000 color images (32x32 pixels) in 10 classes (airplane, car, bird, etc.)

Trained using a Convolutional Neural Network (CNN)

Installation & Usage

## Requirements

### Ensure you have Python 3.7+ and install the dependencies:
pip install torch torchvision
