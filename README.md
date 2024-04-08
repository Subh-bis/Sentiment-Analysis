# Sentiment-Analysis

Title: Emotion Detection with Convolutional Neural Networks (CNNs)

Description

This project implements a CNN-based emotion detection system using Keras and OpenCV. 
It can classify facial expressions into six categories: angry, fear, happy, neutral, sad, and surprise.

Features

Emotion classification using a pre-trained CNN model
Real-time emotion detection from webcam
Facial detection using Haar cascade classifier (optional)
Requirements

Python 3 (tested with 3.x versions)
OpenCV library (pip install opencv-python)
Keras and its dependencies (pip install keras)
NumPy (pip install numpy)
Pandas (pip install pandas)
tqdm (pip install tqdm)
Installation

Clone this repository to your local machine.

Open a terminal or command prompt and navigate to the repository directory.

Install the required libraries:

Bash
pip install opencv-python keras numpy pandas tqdm
Use code with caution.
Data Preparation (Optional)

This project assumes you have a pre-existing dataset of labeled facial images for training and testing. You'll need to:

Download or create your own facial image dataset with emotions labeled appropriately.
Organize the dataset into folders named after the corresponding emotions.
Place the dataset folders within the Training and Testing directories of this repository.
Usage (without data preparation)

Run the script directly:

Bash
python emotion_detection.py
Use code with caution.
The script will attempt to load a pre-trained model (emotiondetector.json and emotiondetector.h5) from the current directory. 
If these files are missing, you'll need to train the model (see instructions below).

The webcam will be activated, and the script will detect faces in real-time, classify their expressions using the loaded model, and display the predicted emotions on the screen.

Training a New Model (if data is available)

Ensure you have the dataset prepared in the Training and Testing directories as described above.

Modify the script (emotion_detection.py) to adjust hyperparameters (e.g., number of epochs, learning rate) if needed.

Run the script with the --train flag:

Bash
python emotion_detection.py --train
Use code with caution.
The script will train the CNN model using the provided dataset and save the trained weights (emotiondetector.h5) and model architecture (emotiondetector.json) for future use.

Explanation

The code is well-structured and includes comments to enhance readability. Here's a breakdown of the key steps:

Data Preprocessing (if applicable)

Defines functions to create DataFrames from image paths and labels (createdataframe).
Extracts features (grayscale images) from image paths (extract_features).
Applies Label Encoding to categorical labels (LabelEncoder).
Converts labels to one-hot encoded vectors (to_categorical).
Model Architecture

Constructs a sequential CNN model with:
Convolutional layers with ReLU activation for feature extraction.
Max pooling layers for downsampling.
Dropout layers for regularization.
Flatten layer to convert 2D feature maps to 1D vectors.
Dense layers with ReLU activation for classification.
Softmax output layer for probability distribution of emotions.
Training (if applicable)

Compiles the model with Adam optimizer, categorical crossentropy loss, and accuracy metric.
Trains the model on the prepared training data with validation on the testing data.
Saves the trained model weights (emotiondetector.h5) and architecture (emotiondetector.json).
Real-Time Emotion Detection

Loads the pre-trained model weights and architecture.
Initializes webcam video capture.
Performs facial detection using the Haar cascade classifier (optional).
For each detected face:
Extracts the facial region.
Preprocesses the image (grayscale conversion, resizing).
Predicts the emotion using the loaded model.
Displays the predicted emotion on the frame.
Shows the webcam video with detected faces and predicted emotions.
Additional Notes

You can experiment with different CNN architectures, hyperparameters, and data augmentation techniques to improve the model's performance.
Explore other facial landmark detection techniques for more precise emotion analysis.
Consider using emotion recognition for various applications such as human-computer interaction or sentiment analysis.
