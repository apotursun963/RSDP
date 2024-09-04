Ricania Simulans Detection using CNN
-------------------------------------

Project Overview

This project demonstrates the use of Convolutional Neural Networks (CNNs) to detect vampire butterflies from images. The project involves data preprocessing, model training, and object detection. It uses a dataset of positive (vampire butterflies) and negative (other butterfly species) samples.

Project Structure
-
Dataset Preparation

The dataset folder contains subfolders for positive and negative images.
Positive images are vampire butterflies.
Negative images are other butterfly species.
Data preprocessing scripts resize, normalize, and augment the images.

Model Training

The model_creation script defines and trains a CNN model using the preprocessed images.
The CNN architecture consists of multiple convolutional layers, max pooling, dropout, and dense layers.

Testing and Inference

The vampire_testing script uses the trained model to classify new images and video frames.
It includes functionality for real-time object detection and video processing.
