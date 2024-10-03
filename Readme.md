# Hand Gesture Recognition with CNN

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify hand gestures captured through a webcam. The model is trained on images of different hand gestures, and it can predict gestures in real-time using a webcam feed. The application uses the `cvzone` library for hand detection and a custom-trained CNN for gesture classification.

### Features:

- Real-time hand gesture recognition
- Support for custom gesture datasets
- Simple and efficient CNN model for classification
- Image preprocessing and augmentation for better accuracy
- Uses OpenCV for video capture and display

---

## Folder Structure

- **Model**: Contains the trained CNN model (`keras_model.h5`) and labels (`labels.txt`).
- **Data**: Folder where hand gesture images are stored for training and testing.
  - Example: `Data/Smile`, `Data/ThumbsUp`, etc.

---

### Main Dependencies:

- OpenCV
- cvzone
- TensorFlow / Keras
- NumPy

---

## Usage

### 1. Collect Data:

Use the provided script to capture images for each gesture:

```python
python collect_images.py
```

Press 's' to save an image, and 'ESC' to exit the program. Update the `folder` variable with the path to save gesture images.

### 2. Train the Model:

After collecting the data, use the following script to train the CNN model on your gesture dataset:

```python
python train_model.py
```

### 3. Real-time Gesture Recognition:

Once the model is trained, use the following script for real-time gesture prediction via webcam:

```python
python gesture_recognition.py
```

### 4. Exit the Program:

The program will terminate when you press the 'Stop' gesture or the ESC key.

---

## Acknowledgments

This project utilizes the `cvzone` library for hand detection and the Keras API to build and train the CNN model.
